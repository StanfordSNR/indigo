#!/usr/bin/env python

import argparse
from sender import Sender


def sample_action(state):
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.port)

    sender.set_sample_action(sample_action)

    try:
        sender.run()
    except KeyboardInterrupt:
        sender.clean_up()


if __name__ == '__main__':
    main()
