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

        self.session = tf.Session()

        if self.training:
            # epsilon-greedy exploration probability
            self.explore_prob = 1.0
            self.init_explore_prob = 1.0
            self.final_explore_prob = 0.0
            self.anneal_steps = 50

            self.train_iter = 0
            self.reward_discount = 0.99
            self.learning_rate = 0.02

            self.state_buf_batch = []
            self.action_buf_batch = []
            self.reward_buf_batch = []

            # reward history for normalization
            self.reward_history = MeanVarHistory()

            self.build_tf_graph()
        else:
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.session, self.model_path)
            self.state = tf.get_collection('state')[0]
            self.predicted_action = tf.get_collection('predicted_action')[0]

    def build_tf_graph(self):
        self.build_policy()
        self.build_loss()

        # initialize variables
        self.session.run(tf.global_variables_initializer())

    def build_policy(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        self.W = tf.get_variable('W', [self.state_dim, self.action_cnt],
                                 initializer=tf.random_normal_initializer())
        self.b = tf.get_variable('b', [self.action_cnt],
                                 initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(self.state, self.W) + self.b

        self.predicted_action = tf.reshape(tf.multinomial(
                                           self.action_scores, 1), [])

    def build_loss(self):
        self.taken_action = tf.placeholder(tf.int32, (None,))
        self.discounted_reward = tf.placeholder(tf.float32, (None,))

        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.taken_action, logits=self.action_scores)
        ce_loss *= self.discounted_reward  # core of policy gradient
        ce_loss = tf.reduce_mean(ce_loss)

        reg_penalty = 0.001
        reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            reg_loss += tf.nn.l2_loss(x)
        reg_loss *= reg_penalty

        loss = ce_loss + reg_loss

        optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def sample_action(self, state):
        # epsilon-greedy exploration
        if self.training and np.random.random() < self.explore_prob:
            action = np.random.randint(0, self.action_cnt)
        else:
            norm_state = np.array([state]) / 125.0 - 1
            action = self.session.run(self.predicted_action,
                                      {self.state: norm_state})

        return action

    def anneal_exploration(self):
        ratio = float(self.anneal_steps - self.train_iter) / self.anneal_steps
        ratio = max(ratio, 0)
        self.explore_prob = self.final_explore_prob + ratio * (
                            self.init_explore_prob - self.final_explore_prob)

    def compute_discounted_rewards(self, final_reward, T):
        reward_buf = np.zeros(T)
        reward_buf[T - 1] = final_reward
        for t in reversed(xrange(T - 1)):
            reward_buf[t] = self.reward_discount * reward_buf[t + 1]

        # update reward history for normalization
        self.reward_history.append(reward_buf)
        reward_mean, reward_var = self.reward_history.get_mean_var()

        return (reward_buf - reward_mean) / np.sqrt(reward_var)

    def store_episode(self, state_buf, action_buf, final_reward):
        assert len(state_buf) == len(action_buf)
        T = len(state_buf)
        reward_buf = self.compute_discounted_rewards(final_reward, T)

        self.state_buf_batch.extend(state_buf)
        self.action_buf_batch.extend(action_buf)
        self.reward_buf_batch.extend(reward_buf)

    def update_model(self):
        sys.stderr.write('Updating model...\n')

        norm_state_buf_batch = np.array(self.state_buf_batch) / 125.0 - 1
        self.session.run(self.train_op, {
            self.state: norm_state_buf_batch,
            self.taken_action: self.action_buf_batch,
            self.discounted_reward: self.reward_buf_batch
        })

        self.anneal_exploration()
        self.train_iter += 1

        self.state_buf_batch = []
        self.action_buf_batch = []
        self.reward_buf_batch = []

    def save_model(self):
        saver = tf.train.Saver()
        tf.add_to_collection('state', self.state)
        tf.add_to_collection('predicted_action', self.predicted_action)
        tf.add_to_collection('W', self.W)
        tf.add_to_collection('b', self.b)
        tf.add_to_collection('action_scores', self.action_scores)
        saver.save(self.session, self.model_path)

        sys.stderr.write('\nModel saved to %s\n' % self.model_path)
