#!/usr/bin/env python

import argparse
import project_root
import numpy as np
import tensorflow as tf
from os import path
from env.sender import Sender
from models import ActorCriticLSTM
from a3c import normalize_states


class Learner(object):
    def __init__(self, state_dim, action_cnt, restore_vars):
        with tf.variable_scope('local'):
            self.pi = ActorCriticLSTM(
                state_dim=state_dim, action_cnt=action_cnt)
            # save the current LSTM state of local network
            self.lstm_state = self.pi.lstm_state_init

        self.session = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.pi.trainable_vars)
        saver.restore(self.session, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables()) - set(self.pi.trainable_vars)
        self.session.run(tf.variables_initializer(uninit_vars))

    def sample_action(self, state):
        norm_state = normalize_states([state])

        ops_to_run = [self.pi.action_probs, self.pi.lstm_state_out]
        feed_dict = {
            self.pi.states: norm_state,
            self.pi.lstm_state_in: self.lstm_state,
        }

        ret = self.session.run(ops_to_run, feed_dict)
        action_probs, lstm_state_out = ret

        action = np.argmax(action_probs[0])
        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        self.lstm_state = lstm_state_out
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
