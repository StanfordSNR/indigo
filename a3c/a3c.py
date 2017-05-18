import sys
import time
import project_root
import numpy as np
import tensorflow as tf
from os import path
from models import ActorCriticLSTM
from helpers.helpers import make_sure_path_exists


def normalize_states(states):
    norm_states = np.array(states, dtype=np.float32)

    # queuing_delay, target range [0, 200]
    queuing_delay = norm_states[:, 0]
    queuing_delay /= 100.0
    queuing_delay -= 1.0

    # send_interval and recv_interval, target range [0, 100]
    for i in [1, 2]:
        interval = norm_states[:, i]
        interval /= 50.0
        interval -= 1.0

    # cwnd, target range [0, 200]
    cwnd = norm_states[:, 3]
    cwnd /= 100.0
    cwnd -= 1.0

    # make sure all features lie in [-1.0, 1.0]
    norm_states[norm_states > 1.0] = 1.0
    norm_states[norm_states < -1.0] = -1.0
    return norm_states


class A3C(object):
    def __init__(self, cluster, server, task_index, env):
        # distributed tensorflow related
        self.cluster = cluster
        self.server = server
        self.task_index = task_index
        self.env = env

        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt
        self.worker_device = '/job:worker/task:%d' % task_index
        self.gamma = 0.999

        # step counters
        self.max_global_step = 4000
        self.local_step = 0

        # must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # build tensorflow computation graph
        self.build_tf_graph()

        # summary related
        if self.task_index == 0:
            self.logdir = path.join(project_root.DIR, 'a3c', 'logs')
            make_sure_path_exists(self.logdir)
            self.summary_writer = tf.summary.FileWriter(self.logdir)

        # create session
        self.session = tf.Session(self.server.target)
        self.session.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()

    def build_tf_graph(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                cluster=self.cluster)):
            with tf.variable_scope('global'):
                self.global_network = ActorCriticLSTM(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32,
                    initializer=tf.constant_initializer(0, tf.int32),
                    trainable=False)

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = ActorCriticLSTM(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)

            self.build_loss()

    def build_loss(self):
        pi = self.local_network

        self.actions = tf.placeholder(tf.int32, [None])
        self.rewards = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])

        # policy loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi.action_scores, labels=self.actions)
        policy_loss = tf.reduce_mean(cross_entropy_loss * self.advantages)

        # value loss
        value_loss = 0.5 * tf.reduce_mean(tf.square(
            self.rewards - pi.state_values))

        # add entropy to loss to encourage exploration
        log_action_probs = tf.log(pi.action_probs)
        entropy = -tf.reduce_mean(pi.action_probs * log_action_probs)

        # total loss and gradients
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        grads = tf.gradients(loss, pi.trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 10.0)

        # sync local network to global network
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            pi.trainable_vars, self.global_network.trainable_vars)])

        # calculate gradients and apply to global network
        grads_and_vars = list(zip(grads, self.global_network.trainable_vars))
        inc_global_step = self.global_step.assign_add(1)

        optimizer = tf.train.AdamOptimizer(1e-4)
        self.train_op = tf.group(
            optimizer.apply_gradients(grads_and_vars), inc_global_step)

        # summary related
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('entropy', entropy)
        tf.summary.scalar('total_loss', loss)
        tf.summary.scalar('final_reward', self.rewards[-1])
        tf.summary.scalar('mean_reward', tf.reduce_mean(self.rewards))
        tf.summary.scalar('grad_global_norm', tf.global_norm(grads))
        tf.summary.scalar('var_global_norm', tf.global_norm(pi.trainable_vars))
        self.summary_op = tf.summary.merge_all()

    def sample_action(self, state):
        pi = self.local_network

        # normalize a state
        norm_state = normalize_states([state])

        # run ops in local networks
        ops_to_run = [pi.action_probs, pi.state_values, pi.lstm_state_out]
        feed_dict = {
            pi.states: norm_state,
            pi.lstm_state_in: self.lstm_state,
        }

        ret = self.session.run(ops_to_run, feed_dict)
        action_probs, state_values, lstm_state_out = ret

        # choose an action to take and update current LSTM state
        action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        self.lstm_state = lstm_state_out

        # append state, action, value to episode buffer
        self.state_buf.extend(norm_state)
        self.action_buf.append(action)
        self.value_buf.extend(state_values)

        return action

    def save_model(self, check_point=None):
        if check_point is None:
            time.sleep(5)
            model_path = path.join(self.logdir, 'model')
        else:
            model_path = path.join(self.logdir, 'checkpoint-%d' % check_point)

        # copy global parameters to local
        self.session.run(self.sync_op)

        # save local parameters to worker-0
        saver = tf.train.Saver(self.local_network.trainable_vars)
        saver.save(self.session, model_path)
        sys.stderr.write('\nModel saved to worker-0:%s\n' % model_path)

    def rollout(self):
        # reset environment
        self.env.reset()

        # reset buffers for states, actions and values, and LSTM state
        self.state_buf = []
        self.action_buf = []
        self.value_buf = []
        self.lstm_state = self.local_network.lstm_state_init

        # get an episode of rollout
        final_reward = self.env.rollout()

        # state_buf, action_buf, value_buf should have been filled in
        episode_len = len(self.state_buf)
        assert len(self.action_buf) == episode_len
        assert len(self.value_buf) == episode_len

        # compute discounted returns
        if self.gamma == 1.0:
            self.reward_buf = np.full(episode_len, final_reward)
        else:
            self.reward_buf = np.zeros(episode_len)
            self.reward_buf[-1] = final_reward
            for i in reversed(xrange(episode_len - 1)):
                self.reward_buf[i] = self.reward_buf[i + 1] * self.gamma

        # compute advantages
        self.adv_buf = self.reward_buf - np.asarray(self.value_buf)

    def run(self):
        pi = self.local_network

        global_step = 0
        check_point = 1000
        while global_step < self.max_global_step:
            sys.stderr.write('Global step: %d\n' % global_step)

            # reset local parameters to global
            self.session.run(self.sync_op)

            # get an episode of rollout
            self.rollout()

            # train using the rollout
            summarize = self.task_index == 0 and self.local_step % 10 == 0

            if summarize:
                ops_to_run = [self.train_op, self.global_step, self.summary_op]
            else:
                ops_to_run = [self.train_op, self.global_step]

            ret = self.session.run(ops_to_run, {
                pi.states: self.state_buf,
                self.actions: self.action_buf,
                self.rewards: self.reward_buf,
                self.advantages: self.adv_buf,
                pi.lstm_state_in: pi.lstm_state_init,
            })

            global_step = ret[1]
            self.local_step += 1

            if summarize:
                self.summary_writer.add_summary(ret[2], global_step)
                self.summary_writer.flush()

            if self.task_index == 0 and global_step >= check_point:
                with tf.device(self.worker_device):
                    self.save_model(check_point)
                check_point += 1000

        if self.task_index == 0:
            with tf.device(self.worker_device):
                self.save_model()
