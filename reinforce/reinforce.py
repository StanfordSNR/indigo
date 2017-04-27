import sys
import numpy as np
import tensorflow as tf
import project_root
from helpers.helpers import MeanVarHistory, make_sure_path_exists


class Reinforce(object):
    def __init__(self, state_dim, action_cnt, train=False,
                 save_vars=None, restore_vars=None, debug=False):
        self.state_dim = state_dim
        self.action_cnt = action_cnt
        self.train = train
        self.save_vars = save_vars
        self.restore_vars = restore_vars
        self.debug = debug

        # start tensorflow session and build tensorflow graph
        self.session = tf.Session()
        self.build_tf_graph()
        self.train_iter = 0

        if self.debug:
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)

        if not self.train:  # production
            assert self.save_vars is None
            assert self.restore_vars is not None
        else:
            # buffers for each batch
            self.state_buf_batch = []
            self.action_buf_batch = []

            # reward calculation
            self.reward_buf_batch = []
            self.reward_discount = 1.0
            self.reward_history = MeanVarHistory()

            # epsilon-greedy exploration with decaying probability
            self.explore_prob = 0.5
            self.init_explore_prob = 0.5
            self.final_explore_prob = 0.0
            self.decay_steps = 2000
            self.decay_iter = 0

        if self.restore_vars is None:
            # initialize variables
            self.session.run(tf.global_variables_initializer())
        else:
            # restore saved variables
            saver = tf.train.Saver(self.trainable_vars)
            saver.restore(self.session, self.restore_vars)
            if self.debug:
                print 'Restored variables:'
                print self.session.run(self.trainable_vars)

            # init the remaining vars, especially those created by optimizer
            uninit_vars = set(tf.global_variables()) - set(self.trainable_vars)
            self.session.run(tf.variables_initializer(uninit_vars))

    def build_tf_graph(self):
        self.build_policy()

        if self.train:
            self.build_loss()

        if self.debug:
            summary_path = 'reinforce_summary'
            make_sure_path_exists(summary_path)
            self.summary_writer = tf.summary.FileWriter(
                summary_path, graph=self.session.graph)

            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('policy_loss', self.policy_loss)
            self.summary_op = tf.summary.merge_all()

    def build_policy(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])

        # softmax classification
        h1_dim = 20
        W1 = tf.get_variable('W1', [self.state_dim, h1_dim],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [h1_dim],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.tanh(tf.matmul(self.state, W1) + b1)

        W2 = tf.get_variable('W2', [h1_dim, self.action_cnt],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(h1, W2) + b2
        self.predicted_action = tf.reshape(
            tf.multinomial(self.action_scores, 1), [])

        self.trainable_vars = [W1, b1, W2, b2]

    def build_loss(self):
        self.taken_action = tf.placeholder(tf.int32, [None])

        # regularization loss
        reg_penalty = 0.01
        self.reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            self.reg_loss += tf.nn.l2_loss(x)
        self.reg_loss *= reg_penalty

        # cross entropy loss
        self.ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.taken_action, logits=self.action_scores)
        self.policy_loss = tf.reduce_mean(self.ce_loss)

        # total loss
        self.loss = self.ce_loss + self.reg_loss

        # magic of policy gradient
        self.discounted_reward = tf.placeholder(tf.float32, [None])
        self.loss *= self.discounted_reward

        self.loss = tf.reduce_mean(self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)

    def normalize_state(self, state):
        norm_state = np.array(state, dtype=np.float32) / 100.0 - 1.0

        # queuing_delay, mostly in [0, 210]
        queuing_delays = norm_state[:, 0]
        queuing_delays /= 105.0
        queuing_delays -= 1.0

        # send_ewma and ack_ewma, mostly in [0, 16]
        for i in [1, 2]:
            ewmas = norm_state[:, i]
            ewmas /= 8.0
            ewmas -= 1.0

        # make sure all features lie in [-1.0, 1.0]
        norm_state[norm_state > 1.0] = 1.0
        norm_state[norm_state < -1.0] = -1.0
        return norm_state

    def sample_action(self, state):
        norm_state = self.normalize_state([state])

        if not self.train:
            action = self.session.run(self.predicted_action,
                                      {self.state: norm_state})
        else:
            if np.random.random() < self.explore_prob:
                # epsilon-greedy exploration
                action = np.random.randint(0, self.action_cnt)
            else:
                action = self.session.run(self.predicted_action,
                                          {self.state: norm_state})

        return action

    def decay_exploration(self):
        self.decay_iter += 1
        ratio = float(self.decay_steps - self.decay_iter) / self.decay_steps
        ratio = max(ratio, 0)
        self.explore_prob = self.final_explore_prob + ratio * (
            self.init_explore_prob - self.final_explore_prob)

    def compute_discounted_rewards(self, final_reward, T):
        reward_buf = np.zeros(T)
        reward_buf[T - 1] = final_reward
        for t in reversed(xrange(T - 1)):
            reward_buf[t] = self.reward_discount * reward_buf[t + 1]

        self.reward_history.append(reward_buf)
        self.reward_history.normalize_inplace(reward_buf)

        return reward_buf

    def store_episode(self, state_buf, action_buf, final_reward):
        assert self.train
        assert len(state_buf) == len(action_buf)

        self.state_buf_batch.extend(state_buf)
        self.action_buf_batch.extend(action_buf)

        reward_buf = self.compute_discounted_rewards(
            final_reward, len(state_buf))
        self.reward_buf_batch.extend(reward_buf)

    def update_model(self):
        assert self.train

        self.train_iter += 1
        sys.stderr.write('Updating model...\n')

        norm_state_buf = self.normalize_state(self.state_buf_batch)

        if not self.debug:
            self.session.run(self.train_op, {
                self.state: norm_state_buf,
                self.taken_action: self.action_buf_batch,
                self.discounted_reward: self.reward_buf_batch
            })
        else:
            _, summary = self.session.run([self.train_op, self.summary_op], {
                self.state: norm_state_buf,
                self.taken_action: self.action_buf_batch,
                self.discounted_reward: self.reward_buf_batch
            })

            self.summary_writer.add_summary(summary, self.train_iter)

        self.state_buf_batch = []
        self.action_buf_batch = []
        self.reward_buf_batch = []

    def save_model(self):
        assert self.train
        assert self.save_vars is not None

        saver = tf.train.Saver(self.trainable_vars)
        saver.save(self.session, self.save_vars)
        sys.stderr.write('\nModel saved to %s\n' % self.save_vars)

        if self.debug:
            print 'Saved variables:'
            print self.session.run(self.trainable_vars)
