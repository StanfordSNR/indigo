#!/usr/bin/env python

import sys
import argparse
import numpy as np
from collections import deque
from sender import Sender
from rl.reinforce import Reinforce


class Trainer(object):
    def __init__(self, args):
        self.ip = args.ip
        self.port = args.port

        self.state_dim = 1000
        self.action_cnt = 2
        self.max_steps = 2000
        self.reward_history = deque(maxlen=100)

        if args.episodes is not None:
            self.max_episodes = args.episodes
        else:
            self.max_episodes = 10000

        if args.algorithm == 'reinforce':
            self.learner = Reinforce(state_dim=self.state_dim,
                                     action_cnt=self.action_cnt)

    def run(self):
        for episode_i in xrange(1, self.max_episodes + 1):
            sys.stderr.write('\nEpisode %s is running...\n' % episode_i)
            sender = Sender(self.ip, self.port)
            sender.init_rl_params(
                state_dim=self.state_dim,
                max_steps=self.max_steps,
                delay_weight=0.5,
                sample_action=self.learner.sample_action)

            try:
                sender.run()
            except KeyboardInterrupt:
                sys.exit(0)
            finally:
                sender.cleanup()

            experience = sender.get_experience()
            self.learner.update_model(experience)

            reward = experience[2]
            self.reward_history.append(reward)
            sys.stderr.write('Reward for this episode: %.3f\n' % reward)
            sys.stderr.write('Average reward for the last 100 episodes: %.3f\n'
                             % np.mean(self.reward_history))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    parser.add_argument('--episodes', metavar='N', type=int,
                        help='maximum episodes to train (default 10000)')
    parser.add_argument(
        '--algorithm', choices=['reinforce'], default='reinforce',
        help='reinforcement learning algorithm to train the sender'
        ' (default REINFORCE)')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
