#!/usr/bin/env python

import time
import argparse
from sender import Sender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    def test_sample_action(state):
        time.sleep(1)
        return 1

    MAX_EPISODES = 10
    STATE_DIM = 10
    ACTION_NUM = 3
    MAX_STEPS = 10
    DELAY_WEIGHT = 0.8
    for episode in xrange(MAX_EPISODES):
        sender = Sender(args.ip, args.port, STATE_DIM, ACTION_NUM,
                        test_sample_action, MAX_STEPS, DELAY_WEIGHT)

        try:
            sender.loop()
        except:
            break
        finally:
            sender.cleanup()

        state_buf, action_buf, reward = sender.get_experience()


if __name__ == '__main__':
    main()
