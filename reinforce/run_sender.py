#!/usr/bin/env python

import os
import argparse
import project_root
from sender import Sender
from dagger.dagger import Dagger
from reinforce.reinforce import Reinforce
from helpers.helpers import make_sure_path_exists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('--algorithm', choices=['dagger', 'reinforce'],
                        required=True)
    args = parser.parse_args()

    sender = Sender(args.port)

    curr_file_path = os.path.dirname(os.path.abspath(__file__))
    saved_models_path = os.path.join(curr_file_path, 'saved_models')
    make_sure_path_exists(saved_models_path)

    if args.algorithm == 'dagger':
        model_path = os.path.join(saved_models_path, 'dagger')

        policer = Dagger(
            state_dim=sender.state_dim,
            action_cnt=sender.action_cnt,
            train=False,
            restore_vars=model_path)
    elif args.algorithm == 'reinforce':
        model_path = os.path.join(saved_models_path, 'reinforce')

        policer = Reinforce(
            state_dim=sender.state_dim,
            action_cnt=sender.action_cnt,
            train=False,
            restore_vars=model_path)

    sender.set_sample_action(policer.sample_action)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
