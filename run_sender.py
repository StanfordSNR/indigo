#!/usr/bin/env python

import sys
import time
import argparse
from sender import Sender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.ip, args.port)

    # for test purposes
    def test_sample_action(state):
        time.sleep(1)
        sys.stderr.write('Test: sampling action and sending 1 packet\n')
        return 1

    sender.setup(
        train=True,
        state_dim=10,
        sample_action=test_sample_action,
        max_steps=10,
        delay_weight=0.8)

    try:
        sender.run()
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
