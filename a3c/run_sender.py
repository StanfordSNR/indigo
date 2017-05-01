#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from a3c import ActorCriticNetwork


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars=None):
        with tf.variable_scope('global'):
            self.pi = ActorCriticNetwork(state_dim, action_cnt)

        self.session = tf.Session()

        if restore_vars is None:
            self.session.run(tf.global_variables_initializer())
        else:
            # restore saved variables
            saver = tf.train.Saver(self.pi.trainable_vars)
            saver.restore(self.session, restore_vars)

            # init the remaining vars, especially those created by optimizer
            self.session.run(tf.variables_initializer(
                set(tf.global_variables()) - set(self.pi.trainable_vars)))

    def sample_action(self, state):
        norm_state = self.normalize_states([state])

        action_probs = self.session.run(self.pi.action_probs,
                                        {self.pi.states: norm_state})[0]
        action = np.argmax(np.random.multinomial(1, action_probs - 1e-5))
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
