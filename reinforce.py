import sys
import time
import tensorflow as tf
import numpy as np
from helpers import MeanVarHistory


class Reinforce(object):
    def __init__(self, **params):
        self.training = params['training']
        self.state_dim = params['state_dim']
        self.action_cnt = params['action_cnt']
        self.model_path = params['model_path']
        self.debug = params['debug'] if 'debug' in params else False

        self.session = tf.Session()

        self.max_delay = 250

        if self.debug:
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)

        if self.training:
            # epsilon-greedy exploration with decaying probability
            self.explore_prob = 0.5
            self.init_explore_prob = 0.5
            self.final_explore_prob = 0.0
            self.decay_steps = 100
            self.train_iter = 0
            self.delay_visited_times = np.zeros(2)

            # reward calculation
            self.reward_discount = 0.99
            self.reward_history = MeanVarHistory()

            # buffers for each batch
            self.state_buf_batch = []
            self.action_buf_batch = []
            self.reward_buf_batch = []

            self.build_tf_graph()
        else:
            # restore the trained model if not training
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.session, self.model_path)
            self.state = tf.get_collection('state')[0]
            self.predicted_action = tf.get_collection('predicted_action')[0]
            self.W = tf.get_collection('W')[0]
            self.b = tf.get_collection('b')[0]

            if self.debug:
                print 'Restored W:', self.session.run(self.W)
                print 'Restored b:', self.session.run(self.b)

    def build_tf_graph(self):
        self.build_policy()
        self.build_loss()

        # initialize variables
        self.session.run(tf.global_variables_initializer())

    def build_policy(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])

        # softmax classification
        self.W = tf.get_variable('W', [self.state_dim, self.action_cnt],
                                 initializer=tf.random_normal_initializer())
        self.b = tf.get_variable('b', [self.action_cnt],
                                 initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(self.state, self.W) + self.b

        self.predicted_action = tf.reshape(
                tf.multinomial(self.action_scores, 1), [])

    def build_loss(self):
        self.taken_action = tf.placeholder(tf.int32, [None,])
        self.discounted_reward = tf.placeholder(tf.float32, [None,])

        # regularization loss
        reg_penalty = 0.01
        reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            reg_loss += tf.nn.l2_loss(x)
        reg_loss *= reg_penalty

        # cross entropy loss
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.taken_action, logits=self.action_scores)

        # total loss
        loss = ce_loss + reg_loss
        loss *= self.discounted_reward  # core idea of policy gradient
        loss = tf.reduce_mean(loss)

        if self.debug:
            self.reg_loss = reg_loss
            self.loss = loss

        # decaying learning rate
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1.0
        learning_rate = tf.train.exponential_decay(
                starter_learning_rate, global_step, 10, 0.9, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        self.train_op = optimizer.minimize(loss, global_step=global_step)

    def normalize_state_inplace(self, state):
         # rescale to [-1, 1]
        state /= self.max_delay / 2.0
        state -= 1.0
        state[state > 1] = 1

    def sample_action(self, state):
        assert self.action_cnt == 3

        if self.training:
            delay= min(int(state[0]), self.max_delay)
            if delay < self.max_delay / 2.0:
                self.delay_visited_times[0] += 1
            else:
                self.delay_visited_times[1] += 1

        # epsilon-greedy exploration
        if self.training and np.random.random() < self.explore_prob:
            if self.delay_visited_times[0] * 2 < self.delay_visited_times[1]:
                action = 0
            elif self.delay_visited_times[0] > 2 * self.delay_visited_times[1]:
                action = 2
            else:
                action = np.random.randint(0, self.action_cnt)
        else:
            norm_state = np.array([state], dtype=np.float64)
            self.normalize_state_inplace(norm_state)
            action = self.session.run(self.predicted_action,
                                      {self.state: norm_state})

        return action

    def decay_exploration(self):
        ratio = float(self.decay_steps - self.train_iter) / self.decay_steps
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
        assert len(state_buf) == len(action_buf)
        T = len(state_buf)
        reward_buf = self.compute_discounted_rewards(final_reward, T)

        self.state_buf_batch.extend(state_buf)
        self.action_buf_batch.extend(action_buf)
        self.reward_buf_batch.extend(reward_buf)

    def update_model(self):
        sys.stderr.write('Updating model...\n')

        norm_state_buf_batch = np.array(self.state_buf_batch, dtype=np.float64)
        self.normalize_state_inplace(norm_state_buf_batch)

        self.session.run(self.train_op, {
            self.state: norm_state_buf_batch,
            self.taken_action: self.action_buf_batch,
            self.discounted_reward: self.reward_buf_batch
        })

        if self.debug:
            print 'W:', self.session.run(self.W)
            print 'b:', self.session.run(self.b)
            print 'regularization loss:', self.session.run(self.reg_loss)
            print 'total loss:', self.session.run(self.loss, {
                self.state: norm_state_buf_batch,
                self.taken_action: self.action_buf_batch,
                self.discounted_reward: self.reward_buf_batch
            })

        self.train_iter += 1
        self.decay_exploration()

        self.state_buf_batch = []
        self.action_buf_batch = []
        self.reward_buf_batch = []

    def save_model(self):
        saver = tf.train.Saver()
        tf.add_to_collection('state', self.state)
        tf.add_to_collection('predicted_action', self.predicted_action)
        tf.add_to_collection('W', self.W)
        tf.add_to_collection('b', self.b)

        saver.save(self.session, self.model_path)
        sys.stderr.write('\nModel saved to %s\n' % self.model_path)
