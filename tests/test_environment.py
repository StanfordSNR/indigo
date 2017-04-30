#!/usr/bin/env python

import sys
import project_root
import numpy as np
from os import path
from env.environment import Environment


def create_env():
    uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
    downlink_trace = uplink_trace
    mahimahi_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--downlink-queue=droptail --downlink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mahimahi_cmd)
    env.setup()
    return env


class Learner(object):
    def __init__(self, env):
        self.env = env

        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt
        env.set_sample_action(self.sample_action)

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)

    def cleanup(self):
        self.env.cleanup()

    def run(self):
        for episode_i in xrange(1, 4):
            sys.stderr.write('\nEpisode %d\n' % episode_i)

            # get an episode of experience
            rollout = self.env.rollout()

            # update model
            self.update_model(rollout)

    def update_model(self, rollout):
        sys.stderr.write('Updating model...\n')


def main():
    env = create_env()
    learner = Learner(env)
    try:
        learner.run()
    except KeyboardInterrupt:
        pass
    finally:
        learner.cleanup()


if __name__ == '__main__':
    main()
