#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
from sender import Sender
from reinforce import Reinforce
from helpers import RingBuffer


class Trainer(object):
    def __init__(self, args):
        self.sender = Sender(args.ip, args.port)

        self.state_dim = 500
        self.action_cnt = 2
        self.max_steps = 3000
        self.reward_history = RingBuffer(100)

        if args.episodes is not None:
            self.max_episodes = args.episodes
        else:
            self.max_episodes = 1000

        model_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_path, 'saved_models/rlcc-model')
        if args.algorithm == 'reinforce':
            self.learner = Reinforce(
                train=True,
                state_dim=self.state_dim,
                action_cnt=self.action_cnt,
                model_path=model_path)

        self.sender.setup(
            train=True,
            state_dim=self.state_dim,
            sample_action=self.learner.sample_action,
            max_steps=self.max_steps,
            delay_weight=0.5)

    def run(self):
        for episode_i in xrange(1, self.max_episodes + 1):
            sys.stderr.write('\nEpisode %s is running...\n' % episode_i)

            self.sender.run()
            experience = self.sender.get_experience()
            self.learner.update_model(experience)
            self.sender.reset()

            reward = experience[2]
            self.reward_history.append(reward)
            sys.stderr.write('Reward for this episode: %.3f\n' % reward)
            sys.stderr.write('Average reward for the last 100 episodes: %.3f\n'
                             % np.mean(self.reward_history.get()))

        self.learner.save_model()

    def cleanup(self):
        self.sender.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    parser.add_argument('--episodes', metavar='N', type=int,
                        help='maximum episodes to train (default 100)')
    parser.add_argument(
        '--algorithm', choices=['reinforce'], default='reinforce',
        help='reinforcement learning algorithm to train the sender'
        ' (default REINFORCE)')
    args = parser.parse_args()

    trainer = Trainer(args)
    try:
        trainer.run()
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()
