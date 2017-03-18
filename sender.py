#!/usr/bin/env python

import sys
import time
import json
import socket
import argparse
import numpy as np
from helpers import curr_ts_ms


class Sender(object):
    def __init__(self,
                 ip,
                 port,
                 state_dim,
                 action_num,
                 sample_action,  # sample_action(state) is a function
                 max_steps,
                 delay_weight):
        self.dest_addr = (ip, port)
        self.state_dim = state_dim
        self.action_num = action_num
        self.sample_action = sample_action
        self.max_steps = max_steps
        self.delay_weight = delay_weight

        self.rtts = []
        self.state_buf = []
        self.action_buf = []
        self.acked_bytes = 0

    def get_curr_state(self):
        if len(self.rtts) < self.state_dim:
            return [0] * (self.state_dim - len(self.rtts)) + self.rtts
        else:
            return self.rtts[-self.state_dim:]

    def compute_reward(self):
        delay_percentile = np.percentile(self.rtts, 95) * 0.001
        avg_throughput = self.acked_bytes * 8 * 0.001 / self.runtime
        self.reward = np.log(avg_throughput) - (self.delay_weight *
                                                np.log(delay_percentile))

    def loop(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        data = {}
        data['payload'] = 'x' * 1400

        start_ts = curr_ts_ms()
        for t in xrange(self.max_steps):
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
            self.acked_bytes += ack['acked_bytes']

        self.runtime = curr_ts_ms() - start_ts
        self.compute_reward()

    def get_experience(self):
        return self.state_buf, self.action_buf, self.reward

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

    sender = Sender(args.ip, args.port, 10, 3, test_sample_action, 10, 0.8)
    try:
        sender.loop()
    except:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
