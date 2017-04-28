#!/usr/bin/env python

import argparse
from receiver import Receiver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    receiver = Receiver(args.ip, args.port)

    try:
        receiver.handshake()
        receiver.run()
    except KeyboardInterrupt:
        pass
    finally:
        receiver.cleanup()


if __name__ == '__main__':
    main()
