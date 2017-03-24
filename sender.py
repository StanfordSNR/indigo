#!/usr/bin/env python

import sys
import time
import json
import socket
import argparse
import numpy as np
from helpers import curr_ts_ms


class Sender(object):
    def __init__(self, ip, port):
        self.dest_addr = (ip, port)

    def init_rl_params(self, **rl_params):
        self.state_dim = rl_params['state_dim']
        self.sample_action = rl_params['sample_action']
        self.max_steps = rl_params['max_steps']
        self.delay_weight = rl_params['delay_weight']

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
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / self.duration
        delay_percentile = float(np.percentile(self.rtts, 95))

        sys.stderr.write('Average throughput: %s Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile RTT: %s ms\n' % delay_percentile)

        self.reward = np.log(max(avg_throughput, 1e-5))
        self.reward -= self.delay_weight * max(
                       np.log(max(0.02 * delay_percentile, 1e-5)), 0)

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        data = {}
        data['payload'] = 'x' * 1400

        first_send_ts = sys.maxint
        last_send_ts = 0

        sys.stderr.write('[')
        progress = 0
        for t in xrange(self.max_steps):
            if float(t) / self.max_steps > progress:
                sys.stderr.write('-')
                progress += 0.1

            state = self.get_curr_state()
            self.state_buf.append(state)

            action = self.sample_action(state)
            self.action_buf.append(action)

            for i in xrange(action):
                data['send_ts'] = curr_ts_ms()
                serialized_data = json.dumps(data)
                s.sendto(serialized_data, self.dest_addr)

            raw_ack = s.recvfrom(1500)[0]
            ack = json.loads(raw_ack)
            send_ts = ack['send_ts']
            rtt = curr_ts_ms() - send_ts
            self.rtts.append(rtt)

            self.acked_bytes += ack['acked_bytes']
            first_send_ts = min(send_ts, first_send_ts)
            last_send_ts = max(send_ts, last_send_ts)

        sys.stderr.write(']\n')
        self.duration = last_send_ts - first_send_ts
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

    sender = Sender(args.ip, args.port)

    # for test purposes
    def test_sample_action(state):
        time.sleep(1)
        sys.stderr.write('Test: sampling action and sending 1 packet\n')
        return 1

    sender.init_rl_params(
        state_dim=10,
        max_steps=10,
        delay_weight=0.8,
        sample_action=test_sample_action)

    try:
        sender.run()
    except KeyboardInterrupt:
        sys.exit(0)
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
