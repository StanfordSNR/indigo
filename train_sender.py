#!/usr/bin/env python

import sys
import argparse
from sender import Sender
from rl.reinforce import Reinforce


class Trainer(object):
    def __init__(self, ip, port, algorithm):
        self.ip = ip
        self.port = port

        self.state_dim = 5000
        self.action_cnt = 2
        self.max_episodes = 300
        self.max_steps = 10000

        if algorithm == 'reinforce':
            self.learner = Reinforce(state_dim=self.state_dim,
                                     action_cnt=self.action_cnt)

    def run(self):
        for episode_i in xrange(1, self.max_episodes + 1):
            sys.stderr.write('\nEpisode %s is running...\n' % episode_i)
            sender = Sender(self.ip, self.port)
            sender.init_rl_params(
                state_dim=self.state_dim,
                max_steps=self.max_steps,
                delay_weight=1.0,
                loss_weight=10.0,
                sample_action=self.learner.sample_action)

            try:
                sender.run()
            except KeyboardInterrupt:
                sys.exit(0)
            finally:
                sender.cleanup()

            experience = sender.get_experience()
            sys.stderr.write('Reward: %s\n' % experience[2])

            self.learner.update_model(experience)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    parser.add_argument(
        '--algorithm', choices=['reinforce'], default='reinforce',
        help='reinforcement learning algorithm to train the sender')
    args = parser.parse_args()

    trainer = Trainer(args.ip, args.port, args.algorithm)
    trainer.run()


if __name__ == '__main__':
    main()
