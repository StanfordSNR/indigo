#!/usr/bin/env python

import time
import argparse
from sender import Sender
from helpers import test_sender_rl_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    MAX_EPISODES = 10
    for episode in xrange(MAX_EPISODES):
        sender = Sender(args.ip, args.port)
        sender.init_rl_params(test_sender_rl_params())

        try:
            sender.loop()
        except:
            break
        finally:
            sender.cleanup()

        state_buf, action_buf, reward = sender.get_experience()


if __name__ == '__main__':
    main()
