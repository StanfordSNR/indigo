#!/usr/bin/env python

import sys
import argparse
import numpy as np
import project_root
from env.sender import Sender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    parser.add_argument('--cwnd', metavar='GUESS', type=int, required=True)
    args = parser.parse_args()

    sender = Sender(args.port)
    sender.set_cwnd(args.cwnd)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
