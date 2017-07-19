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

    def sample_action(self, step_state_buf):
        states = np.asarray(step_state_buf, dtype=np.float32)
        mean_delay = np.mean(states)

        if mean_delay < 30:
            return 0
        elif mean_delay < 50:
            return 1
        elif mean_delay < 100:
            return 3
        else:
            return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)
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
