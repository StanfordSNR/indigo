#!/usr/bin/env python

import socket
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP', type=str)
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = 'test'
    s.sendto(msg, (args.ip, args.port))


if __name__ == '__main__':
    main()
