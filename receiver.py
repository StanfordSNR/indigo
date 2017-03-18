#!/usr/bin/env python

import sys
import json
import socket
from helpers import curr_ts_ms


class Receiver(object):
    def __init__(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('0.0.0.0', 10000))
        ip, port = s.getsockname()
        sys.stderr.write('Listening on port: %s\n' % port)
        self.s = s

    def start(self):
        ack = {}

        while True:
            raw_data, addr = self.s.recvfrom(1500)
            data = json.loads(raw_data)
            ack['send_ts'] = data['send_ts']
            self.s.sendto(json.dumps(ack), addr)

    def stop(self):
        self.s.close()


def main():
    receiver = Receiver()
    receiver.start()


if __name__ == '__main__':
    main()
