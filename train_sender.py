#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
from sender import Sender
from reinforce import Reinforce


class Trainer(object):
    def __init__(self, args):
        self.sender = Sender(args.ip, args.port)

        self.state_dim = self.sender.state_dim()
        self.action_cnt = self.sender.action_cnt()

        model_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_path, 'saved_models/rlcc-model')

        self.learner = Reinforce(
            training=True,
            state_dim=self.state_dim,
            action_cnt=self.action_cnt,
            model_path=model_path)

        self.sender.setup(
            training=True,
            sample_action=self.learner.sample_action)

        self.max_batches = 100
        self.episodes_per_batch = 1

    def run(self):
        for batch_i in xrange(1, self.max_batches + 1):
            sys.stderr.write('\nBatch %s/%s is running...\n\n' %
                             (batch_i, self.max_batches))

            for episode_i in xrange(1, self.episodes_per_batch + 1):
                sys.stderr.write('Episode %s/%s is running...\n' %
                                 (episode_i, self.episodes_per_batch))

                self.sender.run()
                state_buf, action_buf, reward = self.sender.get_experience()
                self.learner.store_episode(state_buf, action_buf, reward)
                self.sender.reset_training()

                sys.stderr.write('Reward for this episode: %.3f\n' % reward)

            self.learner.update_model()

        self.learner.save_model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    trainer = Trainer(args)
    try:
        trainer.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
