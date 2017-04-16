import sys
import numpy as np
import tensorflow as tf


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

        # buffers for each batch
        self.state_buf_batch = []
        self.action_buf_batch = []

        if self.restore_vars is None:
            # initialize variables
            self.session.run(tf.global_variables_initializer())
        else:
            # restore saved variables
            saver = tf.train.Saver()
            saver.restore(self.session, self.restore_vars)
            if self.debug:
                print 'Restored W:', self.session.run(self.W)
                print 'Restored b:', self.session.run(self.b)

    def build_tf_graph(self):
        self.build_policy()

        if self.training:
            self.build_loss()

    def build_policy(self):
        self.state = tf.placeholder(tf.float32, [None, self.state_dim])

        # softmax classification
        self.W = tf.get_variable('W', [self.state_dim, self.action_cnt],
                                 initializer=tf.random_normal_initializer())
        self.b = tf.get_variable('b', [self.action_cnt],
                                 initializer=tf.constant_initializer(0.0))
        self.action_scores = tf.matmul(self.state, self.W) + self.b
        self.predicted_action = tf.argmax(self.action_scores, 1)

        #self.predicted_action = tf.reshape(
        #        tf.multinomial(self.action_scores, 1), [])

    def build_loss(self):
        self.expert_action = tf.placeholder(tf.int32, [None,])

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

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.9)
        self.train_op = optimizer.minimize(self.loss)

    def normalize_state(self, state):
        norm_state = np.array(state, dtype=np.float32) / 100.0 - 1.0
        norm_state[norm_state > 1.0] = 1.0
        norm_state[norm_state < -1.0] = -1.0
        return norm_state

    def sample_action(self, state):
        if self.train_iter > 0:
            norm_state = self.normalize_state([state])
            action = self.session.run(self.predicted_action,
                                      {self.state: norm_state})
        else:
            action = self.sample_expert_action(state)

        return action

    def sample_expert_action(self, state):
        queuing_delay = state[0]

        if queuing_delay <= 30:
            action = 2
        elif queuing_delay >= 100:
            action = 0
        else:
            action = 1

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
        self.session.run(self.train_op, {
            self.state: norm_state_buf,
            self.expert_action: self.action_buf_batch
        })

        if self.debug:
            print 'W:', self.session.run(self.W)
            print 'b:', self.session.run(self.b)
            print 'regularization loss:', self.session.run(self.reg_loss)
            print 'total loss:', self.session.run(self.loss, {
                self.state: norm_state_buf,
                self.expert_action: self.action_buf_batch
            })

        self.state_buf_batch = []
        self.action_buf_batch = []

    def save_model(self):
        assert self.training
        assert self.save_vars is not None

        saver = tf.train.Saver([self.W, self.b])
        saver.save(self.session, self.save_vars)
        sys.stderr.write('\nModel saved to %s\n' % self.save_vars)
