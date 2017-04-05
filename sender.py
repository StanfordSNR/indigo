import sys
import json
import socket
import numpy as np
from helpers import curr_ts_ms, RingBuffer


class Sender(object):
    def __init__(self, ip, port):
        self.dest_addr = (ip, port)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.data = {}
        self.data['payload'] = 'x' * 1400

    # required to be called before running
    def setup(self, **params):
        self.train = params['train']
        self.state_dim = params['state_dim']
        self.sample_action = params['sample_action']

        self.delay_buf = RingBuffer(self.state_dim)

        if self.train:
            self.setup_train(params)

    def setup_train(self, params):
        self.max_steps = params['max_steps']
        self.delay_weight = params['delay_weight']

        self.state_buf = []
        self.action_buf = []

        # for reward computation
        self.acked_bytes = 0
        self.total_delays = []

    def reset(self):
        self.delay_buf.reset()
        self.state_buf = []
        self.action_buf = []
        self.acked_bytes = 0
        self.total_delays = []

    def get_curr_state(self):
        return self.delay_buf.get()

    def compute_reward(self):
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / self.duration
        delay_percentile = float(np.percentile(self.total_delays, 95))

        sys.stderr.write('Average throughput: %s Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile one-way delay: %s ms\n' %
                         delay_percentile)

        self.reward = np.log(max(avg_throughput, 1e-5))
        self.reward -= self.delay_weight * max(
                       np.log(max(0.03 * delay_percentile, 1e-5)), 0)

    def get_experience(self):
        self.compute_reward()
        return self.state_buf, self.action_buf, self.reward

    def run(self):
        if self.train:
            t = 0
            first_ack_ts = sys.maxint
            last_ack_ts = 0

        while True:
            state = self.get_curr_state()
            action = self.sample_action(state)

            for i in xrange(action):
                self.data['send_ts'] = curr_ts_ms()
                serialized_data = json.dumps(self.data)
                self.s.sendto(serialized_data, self.dest_addr)

            serialized_ack = self.s.recvfrom(1500)[0]
            ack = json.loads(serialized_ack)
            send_ts = ack['send_ts']
            ack_ts = ack['ack_ts']
            delay = ack_ts - send_ts
            self.delay_buf.append(delay)

            if self.train:
                self.state_buf.append(state)
                self.action_buf.append(action)

                self.acked_bytes += ack['acked_bytes']
                first_ack_ts = min(ack_ts, first_ack_ts)
                last_ack_ts = max(ack_ts, last_ack_ts)
                self.total_delays.append(delay)

                t += 1
                if t >= self.max_steps:
                    break

        if self.train:
            self.duration = last_ack_ts - first_ack_ts

    def cleanup(self):
        self.s.close()
