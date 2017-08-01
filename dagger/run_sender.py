#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import DaggerNetwork
from helpers.helpers import ewma


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):

        with tf.variable_scope('global'):
            self.model = DaggerNetwork(state_dim=state_dim, action_cnt=action_cnt)

        self.ewma_window = 3        # alpha = 2 / (window + 1)
        self.sess = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, step_state_buf):

        # For ewma delay, only want first component, the one-way delay
        # For the cwnd, try only the most recent cwnd
        owd_buf = np.asarray([state[0] for state in step_state_buf])
        ewma_delay = ewma(owd_buf, self.ewma_window)
        last_cwnd = step_state_buf[-1][1]

        # Get probability of each action from the local network.
        pi = self.model
        action_probs = self.sess.run(pi.action_probs,
                                     feed_dict={pi.states: [[ewma_delay,
                                                             last_cwnd]]})

        action = np.argmax(action_probs[0])
        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        #temperature = 1.0
        #temp_probs = softmax(action_probs[0] / temperature)
        #action = np.argmax(np.random.multinomial(1, temp_probs - 1e-5))
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    model_path = path.join(project_root.DIR, 'dagger', 'logs',
                           '2017-08-01--01-11-44-true-expert-3',
                           'checkpoint-24000')

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
