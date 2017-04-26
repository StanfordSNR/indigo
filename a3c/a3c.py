import sys
import numpy as np
import tensorflow as tf


class A3C(object):
    def __init__(self, cluster, server, worker_device,
                 state_dim, action_cnt, debug=False):
        # distributed tensorflow related
        self.cluster = cluster
        self.server = server
        self.worker_device = worker_device

        self.state_dim = state_dim
        self.action_cnt = action_cnt
        self.debug = debug

        # start tensorflow session and build tensorflow graph
        self.session = tf.Session(server.target)
        self.build_tf_graph()
        self.train_iter = 0

        # initialize variables
        self.session.run(tf.global_variables_initializer())

        if self.debug:
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)

    def build_tf_graph(self):
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                cluster=cluster)):
            with tf.variable_scope('global'):
                pass

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                pass

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)

    def update_model(self, state_buf, action_buf, reward):
        self.train_iter += 1
        sys.stderr.write('Updating model...\n')
