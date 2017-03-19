import time
import tensorflow as tf


class Reinforce(object):
    def __init__(self, **params):
        self.state_dim = params['state_dim']
        self.action_cnt = params['action_cnt']
        self.init_tf_graph()

    def init_tf_graph(self):
        self.session = tf.Session()
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        init = tf.global_variables_initializer()
        self.session.run(init)

    def policy_network(self, state):
        W1 = tf.get_variable('W1', [self.state_dim, 5],
                             initializer=tf.random_normal_initializer())
        b1 = tf.get_variable('b1', [5],
                             initializer=tf.constant_initializer(0.0))
        h1 = tf.nn.relu(tf.matmul(state, W1) + b1)

        W2 = tf.get_variable('W2', [5, self.action_cnt],
                             initializer=tf.random_normal_initializer())
        b2 = tf.get_variable('b2', [self.action_cnt],
                             initializer=tf.constant_initializer(0.0))
        p = tf.matmul(h1, W2) + b2
        return p

    def sample_action(self, state):
        time.sleep(1)
        return 1

    def store_experience(self, experience):
        return

    def update_model(self):
        return
