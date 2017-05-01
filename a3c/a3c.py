import sys
import project_root
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import layers
from helpers.helpers import make_sure_path_exists


class ActorCriticNetwork(object):
    def __init__(self, state_dim, action_cnt):
        self.states = tf.placeholder(tf.float32, [None, state_dim])

        actor_h1 = layers.relu(self.states, 10)
        self.action_scores = layers.linear(actor_h1, action_cnt)
        self.action_probs = tf.nn.softmax(self.action_scores)

        critic_h1 = layers.relu(self.states, 10)
        self.state_values = tf.reshape(layers.linear(critic_h1, 1), [-1])

        self.trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


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

        # must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # step counters
        self.max_global_step = 4000
        self.local_step = 0

        # build tensorflow computation graph
        self.build_tf_graph()

        # summary related
        if self.task_index == 0:
            self.logdir = path.join(
                path.dirname(path.abspath(__file__)), 'logs')
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
                self.global_network = ActorCriticNetwork(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int32,
                    initializer=tf.constant_initializer(0, tf.int32),
                    trainable=False)

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = ActorCriticNetwork(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)

            self.build_loss()

    def build_loss(self):
        pi = self.local_network

        self.states = pi.states
        self.actions = tf.placeholder(tf.int32, [None])
        self.rewards = tf.placeholder(tf.float32, [None])
        self.advantages = self.rewards - pi.state_values

        # policy loss
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pi.action_scores, labels=self.actions)
        policy_loss = tf.reduce_mean(cross_entropy_loss * self.advantages)

        # value loss
        value_loss = 0.5 * tf.reduce_mean(tf.square(self.advantages))

        # add entropy to loss to encourage exploration
        log_action_probs = tf.log(pi.action_probs)
        entropy = -tf.reduce_mean(pi.action_probs * log_action_probs)

        # total loss and gradients
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        grads = tf.gradients(loss, pi.trainable_vars)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)

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
        tf.summary.scalar('grad_global_norm', tf.global_norm(grads))
        tf.summary.scalar('var_global_norm', tf.global_norm(pi.trainable_vars))
        self.summary_op = tf.summary.merge_all()

    def sample_action(self, state):
        norm_state = self.normalize_states([state])

        action_probs = self.session.run(
            self.local_network.action_probs,
            {self.local_network.states: norm_state})[0]
        action = np.argmax(np.random.multinomial(1, action_probs - 1e-5))
        return action

    def normalize_states(self, states):
        norm_states = np.array(states, dtype=np.float32)

        # queuing_delay, mostly in [0, 210]
        queuing_delays = norm_states[:, 0]
        queuing_delays /= 105.0
        queuing_delays -= 1.0

        # send_ewma and ack_ewma, mostly in [0, 32]
        for i in [1, 2]:
            ewmas = norm_states[:, i]
            ewmas /= 16.0
            ewmas -= 1.0

        # make sure all features lie in [-1.0, 1.0]
        norm_states[norm_states > 1.0] = 1.0
        norm_states[norm_states < -1.0] = -1.0
        return norm_states

    def discount_rewards(self, final_reward, episode_len):
        return [final_reward] * episode_len

    def process_rollout(self, rollout):
        state_buf, action_buf, final_reward = rollout

        state_buf = self.normalize_states(state_buf)
        reward_buf = self.discount_rewards(final_reward, len(action_buf))

        return state_buf, action_buf, reward_buf

    def save_model(self):
        saver = tf.train.Saver(self.global_network.trainable_vars)
        model_path = path.join(self.logdir, 'model')
        saver.save(self.session, model_path)
        sys.stderr.write('\nGlobal network saved to %s\n' % model_path)

    def run(self):
        global_step = 0

        while global_step < self.max_global_step:
            sys.stderr.write('Global step: %d\n' % global_step)

            # reset local parameters to global
            self.session.run(self.sync_op)

            # get an episode of rollout and process it
            rollout = self.env.rollout()
            state_buf, action_buf, reward_buf = self.process_rollout(rollout)

            # train using the rollout
            summarize = self.task_index == 0 and self.local_step % 10 == 0

            if summarize:
                ops_to_run = [self.train_op, self.global_step, self.summary_op]
            else:
                ops_to_run = [self.train_op, self.global_step]

            ret = self.session.run(ops_to_run, {
                self.states: state_buf,
                self.actions: action_buf,
                self.rewards: reward_buf,
            })

            global_step = ret[1]
            self.local_step += 1

            if summarize:
                self.summary_writer.add_summary(ret[2], global_step)
                self.summary_writer.flush()

        if self.task_index == 0:
            self.save_model()
