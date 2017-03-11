#!/usr/bin/env python

import sys
import socket


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('localhost', 0))
    ip, port = s.getsockname()
    sys.stderr.write('Listening on %s:%s\n' % (ip, port))

    data, addr = s.recvfrom(1024)
    sys.stderr.write('Received a message "%s" from %s:%s\n'
                     % (data, addr[0], addr[1]))


if __name__ == '__main__':
    main()
