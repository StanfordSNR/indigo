#!/usr/bin/env python

import sys
import argparse
from receiver import Receiver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    receiver = Receiver(args.port)

    try:
        receiver.run()
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        receiver.cleanup()


if __name__ == '__main__':
     main()
