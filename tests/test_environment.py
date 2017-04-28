#!/usr/bin/env python

import sys
import project_root
import numpy as np
from os import path
from env.environment import Environment


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

            # run env, get an episode of experience and reset env
            self.env.run()
            experience = self.env.get_experience()
            self.env.reset()

            # update model
            self.update_model(experience)

    def update_model(self, experience):
        sys.stderr.write('Updating model...\n')


def main():
    uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
    downlink_trace = uplink_trace
    mahimahi_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--uplink-queue=droptail --uplink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mahimahi_cmd)
    env.setup()

    learner = Learner(env)
    try:
        learner.run()
    except KeyboardInterrupt:
        pass
    finally:
        learner.cleanup()


if __name__ == '__main__':
    main()
