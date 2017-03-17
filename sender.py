#!/usr/bin/env python

import json
import socket
import argparse
from helpers import curr_ts_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = {}
    msg['payload'] = 'x' * 1400
    while True:
        msg['send_ts'] = curr_ts_ms()
        s.sendto(json.dumps(msg), (args.ip, args.port))


if __name__ == '__main__':
    main()
