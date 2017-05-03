#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import ActorCriticNetwork


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('local'):
            self.pi = ActorCriticNetwork(state_dim, action_cnt)

        self.session = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.pi.trainable_vars)
        saver.restore(self.session, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables()) - set(self.pi.trainable_vars)
        self.session.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, state):
        norm_state = self.normalize_states([state])

        action_probs = self.session.run(self.pi.action_probs,
                                        {self.pi.states: norm_state})[0]
        action = np.argmax(action_probs)
        return action

    def normalize_states(self, states):
        norm_states = np.array(states, dtype=np.float32)

        # queuing_delay, mostly in [0, 210]
        queuing_delays = norm_states[:, 0]
        queuing_delays /= 105.0
        queuing_delays -= 1.0

        # send_ewma and ack_ewma, mostly in [0, 32]
        for i in [1, 2]:
            ewmas = norm_states[:, i]
            ewmas /= 16.0
            ewmas -= 1.0

        # cwnd, mostly in [0, 100]
        cwnd = norm_states[:, 3]
        cwnd /= 50.0
        cwnd -= 1.0

        # make sure all features lie in [-1.0, 1.0]
        norm_states[norm_states > 1.0] = 1.0
        norm_states[norm_states < -1.0] = -1.0
        return norm_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    restore_vars = path.join(project_root.DIR, 'a3c', 'logs', 'model')

    learner = Learner(
        state_dim=sender.state_dim,
        action_cnt=sender.action_cnt,
        restore_vars=restore_vars)

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
