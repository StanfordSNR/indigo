#!/usr/bin/env python

import sys
import json
import socket
import signal
import argparse
from helpers import curr_ts_ms


EPISODE_TIME = 60  # time (s) of each episode of experience


class Sender(object):
    def __init__(self, dest_addr):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest_addr = dest_addr

    # dummy version
    def get_next_action(self):
        return 1

    def start(self):
        data = {}
        data['payload'] = 'x' * 1400

        while True:
            # determine the next action
            send_cnt = self.get_next_action()
            for i in xrange(send_cnt):
                data['send_ts'] = curr_ts_ms()
                self.s.sendto(json.dumps(data), self.dest_addr)

            # wait for a single ACK
            raw_ack = self.s.recvfrom(1500)[0]
            ack = json.loads(raw_ack)
            rtt = curr_ts_ms() - ack['send_ts']

            # TODO: store RTTs as states


    def stop(self):
        self.s.close()


def timeout_handler(signum, frame):
    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    dest_addr = (args.ip, args.port)
    sender = Sender(dest_addr)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(EPISODE_TIME)

    try:
        sender.start()
    except:
        sys.stderr.write('Finished one episode of experience\n')

    sender.stop()


if __name__ == '__main__':
    main()
