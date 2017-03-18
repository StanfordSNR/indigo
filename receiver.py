#!/usr/bin/env python

import sys
import json
import socket
import argparse


class Receiver(object):
    def __init__(self, port=0):
        self.port = port

    def loop(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', self.port))
        sys.stderr.write('Listening on port: %s\n' % self.port)

        ack = {}
        while True:
            raw_data, addr = s.recvfrom(1500)
            data = json.loads(raw_data)
            ack['send_ts'] = data['send_ts']
            ack['acked_bytes'] = len(raw_data)
            s.sendto(json.dumps(ack), addr)

    def cleanup(self):
        self.s.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    receiver = Receiver(args.port)
    try:
        receiver.loop()
    except:
        pass
    finally:
        receiver.cleanup()


if __name__ == '__main__':
    main()
