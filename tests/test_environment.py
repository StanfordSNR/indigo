#!/usr/bin/env python

import project_root
import numpy as np
from os import path
from env.environment import Environment


class Policy(object):
    def __init__(self, env):
        self.env = env

        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt
        env.set_sample_action(self.sample_action)

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)

    def run(self):
        self.env.run()


def main():
    uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
    downlink_trace = uplink_trace
    mahimahi_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--uplink-queue=droptail --uplink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mahimahi_cmd)
    env.setup()

    policy = Policy(env)
    try:
        policy.run()
    except KeyboardInterrupt:
        pass
    finally:
        env.cleanup()


if __name__ == '__main__':
    main()
