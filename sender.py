#!/usr/bin/env python

import sys
import time
import json
import socket
import argparse
from helpers import curr_ts_ms


class Sender(object):
    def __init__(self, ip, port, sample_action):
        self.dest_addr = (ip, port)
        # sample_action(state) is a function
        self.sample_action = sample_action

        self.state_dim = 10
        self.action_num = 3

        self.rtts = []
        self.state_buf = []
        self.action_buf = []

    def get_state_dim(self):
        return self.state_dim

    def get_action_num(self):
        return self.action_num

    def get_curr_state(self):
        if len(self.rtts) < self.state_dim:
            return [0] * (self.state_dim - len(self.rtts)) + self.rtts
        else:
            return self.rtts[-self.state_dim:]

    def loop(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s

        data = {}
        data['payload'] = 'x' * 1400
        while True:
            state = self.get_curr_state()
            self.state_buf.append(state)

            action = self.sample_action(state)
            self.action_buf.append(action)

            # send #action datagrams
            for i in xrange(action):
                data['send_ts'] = curr_ts_ms()
                s.sendto(json.dumps(data), self.dest_addr)

            raw_ack = s.recvfrom(1500)[0]
            ack = json.loads(raw_ack)
            rtt = curr_ts_ms() - ack['send_ts']
            self.rtts.append(rtt)

    def cleanup(self):
        self.s.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    def test_sample_action(state):
        time.sleep(1)
        return 1

    sender = Sender(args.ip, args.port, test_sample_action)
    try:
        sender.loop()
    except:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
