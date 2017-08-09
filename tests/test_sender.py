#!/usr/bin/env python

import sys
import argparse
import numpy as np
import project_root
from env.sender import Sender


class Policy(object):
    def __init__(self, state_dim, action_cnt):
        self.state_dim = state_dim
        self.action_cnt = action_cnt

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port, train=False)
    policy = Policy(sender.state_dim, sender.action_cnt)
    sender.set_sample_action(policy.sample_action)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
