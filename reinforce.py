import sys
import time
import tensorflow as tf
import numpy as np


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
            self.anneal_steps = 100

            self.train_iter = 0
            self.reward_discount = 0.99
            self.learning_rate = 0.001

            self.state_buf_batch = []
            self.action_buf_batch = []
            self.reward_buf_batch = []

            # reward history for normalization
            self.reward_len = 0
            self.reward_mean = 0.0
            self.reward_square_mean = 0.0

            self.build_tf_graph()
        else:
            saver = tf.train.import_meta_graph(self.model_path + '.meta')
            saver.restore(self.session, self.model_path)
            self.state = tf.get_collection('state')[0]
            self.predicted_action = tf.get_collection('predicted_action')[0]

    def build_tf_graph(self):
        self.build_policy_network()
        self.build_loss()

        # initialize variables
        self.session.run(tf.global_variables_initializer())

    def build_policy_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        W1 = tf.get_variable('W1', [self.state_dim, 20],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [20],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.tanh(tf.matmul(self.state, W1) + b1)

        W2 = tf.get_variable('W2', [20, self.action_cnt], initializer=
                             tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(h1, W2) + b2
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

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss)

    def sample_action(self, state):
        # epsilon-greedy exploration
        if self.training and np.random.random() < self.explore_prob:
            action = np.random.randint(0, 3)
        else:
            state = np.array([state])
            action = self.session.run(self.predicted_action,
                                      {self.state: state})

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
        reward_len_new = self.reward_len + T
        ratio_new = float(T) / reward_len_new
        ratio_old = float(self.reward_len) / reward_len_new

        self.reward_len = reward_len_new
        self.reward_mean = (self.reward_mean * ratio_old +
                            np.mean(reward_buf) * ratio_new)
        self.reward_square_mean = (self.reward_square_mean * ratio_old +
                                   np.mean(np.square(reward_buf)) * ratio_new)

        var = self.reward_square_mean - np.square(self.reward_mean)
        reward_buf -= self.reward_mean
        reward_buf /= np.sqrt(var)

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

        self.session.run(self.train_op, {
            self.state: self.state_buf_batch,
            self.taken_action: self.action_buf_batch,
            self.discounted_reward: self.reward_buf_batch
        })

        self.anneal_exploration()
        self.train_iter += 1

        self.state_buf_batch = []
        self.action_buf_batch = []
        self.reward_buf_batch = []

    def save_model(self):
        tf.add_to_collection('state', self.state)
        tf.add_to_collection('predicted_action', self.predicted_action)
        saver = tf.train.Saver()
        saver.save(self.session, self.model_path)
