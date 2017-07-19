#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import ActorCriticNetwork
from a3c import normalize_state


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('local'):
            self.pi = ActorCriticNetwork(
                state_dim=state_dim, action_cnt=action_cnt)

        self.session = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.pi.trainable_vars)
        saver.restore(self.session, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables()) - set(self.pi.trainable_vars)
        self.session.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, state, cwnd):
        norm_state = normalize_state([state])

        ops_to_run = [self.pi.action_probs]
        feed_dict = {
            self.pi.states: norm_state_buf,
        }

        action_probs = self.session.run(ops_to_run, feed_dict)

        # action = np.argmax(action_probs[0])
        # action = np.argmax(np.random.multinomial(1, action_probs[0][0] - 1e-5))
        temprature = 1.0
        temp_probs = softmax(action_probs[0][0] / temprature)
        action = np.argmax(np.random.multinomial(1, temp_probs - 1e-5))

        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    model_path = path.join(project_root.DIR, 'a3c', 'logs', 'model')

    learner = Learner(
        state_dim=sender.state_dim,
        action_cnt=sender.action_cnt,
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
