#!/usr/bin/env python

import os
import argparse
from sender import Sender
from reinforce import Reinforce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    sender = Sender(args.ip, args.port, training=False)
    state_dim = sender.state_dim()
    action_cnt = sender.action_cnt()

    model_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_path, 'saved_models/rlcc-model')
    policer = Reinforce(
        training=False,
        state_dim=state_dim,
        action_cnt=action_cnt,
        model_path=model_path)

    sender.set_sample_action(policer.sample_action)

    try:
        sender.run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
