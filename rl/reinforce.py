import time
import tensorflow as tf
import numpy as np


class Reinforce(object):
    def __init__(self, **params):
        self.state_dim = params['state_dim']
        self.action_cnt = params['action_cnt']
        self.build_tf_graph()

    def build_tf_graph(self):
        self.session = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.build_policy_network()

        # initialize variables
        self.session.run(tf.global_variables_initializer())

    def build_policy_network(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])
        W1 = tf.get_variable('W1', [self.state_dim, 5],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [5],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(self.state, W1) + b1)

        W2 = tf.get_variable('W2', [5, self.action_cnt],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(h1, W2) + b2
        self.predicted_action = tf.multinomial(self.action_scores, 1)

    def sample_action(self, state):
        state = np.array([state])
        action = self.session.run(self.predicted_action, {self.state: state})
        print 'Will be taking action %s in the future' % action[0][0]

        time.sleep(1)
        return 1

    def compute_discounted_rewards(self):
        return

    def store_experience(self, experience):
        self.compute_discounted_rewards()
        return

    def update_model(self):
        return
