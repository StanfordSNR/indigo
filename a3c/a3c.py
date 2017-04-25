import sys
import numpy as np
import tensorflow as tf


class A3C(object):
    def __init__(self, cluster, server, device, state_dim, action_cnt,
                 save_vars=None, debug=False):
        # distributed tensorflow related
        self.cluster = cluster
        self.server = server
        self.device = device

        self.state_dim = state_dim
        self.action_cnt = action_cnt
        self.save_vars = save_vars
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
        self.trainable_vars = []

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)

    def update_model(self, state_buf, action_buf, reward):
        self.train_iter += 1
        sys.stderr.write('Updating model...\n')

    def save_model(self):
        assert self.save_vars is not None

        saver = tf.train.Saver(self.trainable_vars)
        saver.save(self.session, self.save_vars)
        sys.stderr.write('\nModel saved to %s\n' % self.save_vars)

        if self.debug:
            print 'Saved variables:'
            print self.session.run(self.trainable_vars)
