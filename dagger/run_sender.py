#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import DaggerLSTM
from helpers.helpers import normalize, softmax


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('global'):
            self.model = DaggerLSTM(state_dim=state_dim, action_cnt=action_cnt)

        self.lstm_state = self.model.zero_init_state(1)

        self.sess = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, state):
        state = normalize(state)

        # Get probability of each action from the local network.
        pi = self.model
        feed_dict = {
            pi.input: [[state]],
            pi.state_in: self.lstm_state,
        }
        ops_to_run = [pi.action_probs, pi.state_out]
        action_probs, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take
        action = np.argmax(action_probs[0][0])

        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        # temperature = 1.0
        # temp_probs = softmax(action_probs[0] / temperature)
        # action = np.argmax(np.random.multinomial(1, temp_probs - 1e-5))
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    model_path = path.join(project_root.DIR, 'dagger', 'logs',
                           '2017-08-17--20-59-07', 'checkpoint-100')

    learner = Learner(
        state_dim=Sender.state_dim,
        action_cnt=Sender.action_cnt,
        restore_vars=model_path)

    sender.set_sample_action(learner.sample_action)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
