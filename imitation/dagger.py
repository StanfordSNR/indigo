import sys
import numpy as np
import tensorflow as tf
import project_root
from helpers.helpers import make_sure_path_exists


class Dagger(object):
    def __init__(self, state_dim, action_cnt, training=False,
                 save_vars=None, restore_vars=None, debug=False):
        self.state_dim = state_dim
        self.action_cnt = action_cnt
        self.training = training
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

        if not self.training:  # production
            assert self.save_vars is None
            assert self.restore_vars is not None
        else:
            # buffers for each batch
            self.state_buf_batch = []
            self.action_buf_batch = []

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

        if self.training:
            self.build_loss()

        if self.debug:
            summary_path = 'dagger_summary'
            make_sure_path_exists(summary_path)
            self.summary_writer = tf.summary.FileWriter(
                    summary_path, graph=self.session.graph)

            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('total_loss', self.loss)
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
                tf.argmax(self.action_scores, 1), [])

        self.trainable_vars = [W1, b1, W2, b2]

    def build_loss(self):
        self.expert_action = tf.placeholder(tf.int32, [None, ])

        # regularization loss
        reg_penalty = 0.01
        self.reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            self.reg_loss += tf.nn.l2_loss(x)
        self.reg_loss *= reg_penalty

        # cross entropy loss
        self.ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.expert_action, logits=self.action_scores)

        # total loss
        self.loss = self.ce_loss + self.reg_loss
        self.loss = tf.reduce_mean(self.loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)

    def normalize_state(self, state):
        norm_state = np.array(state, dtype=np.float32)

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

    def sample_expert_action(self, state):
        queuing_delay = state[0]

        for action in xrange(5):
            if queuing_delay >= 10 * (4 - action):
                return action

        return action

    def sample_action(self, state):
        if not self.training or self.train_iter > 0:
            norm_state = self.normalize_state([state])
            action = self.session.run(self.predicted_action,
                                      {self.state: norm_state})
        else:
            action = self.sample_expert_action(state)

        return action

    def store_episode(self, state_buf):
        assert self.training

        self.state_buf_batch.extend(state_buf)

        # label states with expert actions
        for state in state_buf:
            expert_action = self.sample_expert_action(state)
            self.action_buf_batch.append(expert_action)

    def update_model(self):
        assert self.training

        self.train_iter += 1
        sys.stderr.write('Updating model...\n')

        norm_state_buf = self.normalize_state(self.state_buf_batch)

        if not self.debug:
            self.session.run(self.train_op, {
                self.state: norm_state_buf,
                self.expert_action: self.action_buf_batch
            })
        else:
            _, summary = self.session.run([self.train_op, self.summary_op], {
                self.state: norm_state_buf,
                self.expert_action: self.action_buf_batch
            })

            self.summary_writer.add_summary(summary, self.train_iter)

        self.state_buf_batch = []
        self.action_buf_batch = []

    def save_model(self):
        assert self.training
        assert self.save_vars is not None

        saver = tf.train.Saver(self.trainable_vars)
        saver.save(self.session, self.save_vars)
        sys.stderr.write('\nModel saved to %s\n' % self.save_vars)

        if self.debug:
            print 'Saved variables:'
            print self.session.run(self.trainable_vars)
