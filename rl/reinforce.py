import sys
import time
import tensorflow as tf
import numpy as np


class Reinforce(object):
    def __init__(self, **params):
        self.state_dim = params['state_dim']
        self.action_cnt = params['action_cnt']

        self.session = tf.Session()

        # epsilon-greedy exploration probability
        self.explore_prob = 0.5
        self.init_explore_prob = 0.5
        self.final_explore_prob = 0.0
        self.anneal_steps = 1000

        self.train_iter = 0
        self.reward_discount = 0.999

        # reward history for normalization
        self.reward_len = 0
        self.reward_mean = 0.0
        self.reward_var = 1.0

        self.build_tf_graph()

    def build_tf_graph(self):
        self.build_policy_network()
        self.build_loss()
        self.build_gradients()

        # initialize variables
        self.session.run(tf.global_variables_initializer())

    def build_policy_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        W1 = tf.get_variable('W1', [self.state_dim, 20],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [20],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1)

        W2 = tf.get_variable('W2', [20, self.action_cnt], initializer=
                             tf.random_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(h1, W2) + b2
        self.predicted_action = tf.multinomial(self.action_scores, 1)

    def build_loss(self):
        self.taken_action = tf.placeholder(tf.int32, (None,))

        # create nodes to compute cross entropy and regularization loss
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.taken_action, logits=self.action_scores)
        ce_loss = tf.reduce_mean(ce_loss)

        reg_penalty = 0.001
        reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            reg_loss += tf.nn.l2_loss(x)
        reg_loss *= reg_penalty

        self.loss = ce_loss + reg_loss

    def build_gradients(self):
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)

        # create nodes to compute gradients update used in REINFORCE
        self.gradients = self.optimizer.compute_gradients(self.loss)

        self.discounted_reward = tf.placeholder(tf.float32, (None,))
        for i, (grad, var) in enumerate(self.gradients):
            assert grad is not None
            self.gradients[i] = (-self.discounted_reward * grad, var)

        # create nodes to apply gradients
        self.train_op = self.optimizer.apply_gradients(self.gradients)

    def sample_action(self, state):
        # epsilon-greedy exploration
        if np.random.random() < self.explore_prob:
            action = np.random.randint(0, self.action_cnt)
        else:
            state = np.array([state])
            action = self.session.run(self.predicted_action,
                                      {self.state: state})[0][0]

        return action + 1

    def anneal_exploration(self):
        ratio = float(self.anneal_steps - self.train_iter) / self.anneal_steps
        ratio = max(ratio, 0)
        self.explore_prob = self.final_explore_prob + ratio * (
                            self.init_explore_prob - self.final_explore_prob)

    def update_model(self, experience):
        sys.stderr.write('Updating model...\n')

        state_buf, action_buf, final_reward = experience
        assert len(state_buf) == len(action_buf)
        T = len(state_buf)

        # compute discounted rewards
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
        self.reward_var = (self.reward_var * ratio_old +
                           np.var(reward_buf) * ratio_new)

        reward_buf -= self.reward_mean
        reward_buf /= np.sqrt(self.reward_var)

        # update variables in policy network
        for t in xrange(T - 1):
            state = np.array([state_buf[t]])
            action = np.array([action_buf[t]]) - 1
            discounted_reward = np.array([reward_buf[t]])

            self.session.run(self.train_op, {
                self.state: state,
                self.taken_action: action,
                self.discounted_reward: discounted_reward
            })

        self.anneal_exploration()
        self.train_iter += 1
