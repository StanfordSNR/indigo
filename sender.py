import sys
import json
import socket
import numpy as np
from helpers import curr_ts_ms


class Sender(object):
    def __init__(self, ip, port):
        self.dest_addr = (ip, port)

    # required to be called to run
    def setup(self, **params):
        self.train = params['train']
        self.state_dim = params['state_dim']
        self.sample_action = params['sample_action']

        self.delays = []

        if self.train:
            self.setup_train(params)

    def setup_train(self, params):
        self.max_steps = params['max_steps']
        self.delay_weight = params['delay_weight']

        self.acked_bytes = 0

        self.state_buf = []
        self.action_buf = []

        self.first_ack_ts = sys.maxint
        self.last_ack_ts = 0

    def get_curr_state(self):
        if len(self.delays) < self.state_dim:
            return [0] * (self.state_dim - len(self.delays)) + self.delays
        else:
            return self.delays[-self.state_dim:]

    def compute_reward(self):
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / self.duration
        delay_percentile = float(np.percentile(self.delays, 95))

        sys.stderr.write('Average throughput: %s Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile one-way delay: %s ms\n' %
                         delay_percentile)

        self.reward = np.log(max(avg_throughput, 1e-5))
        self.reward -= self.delay_weight * max(
                       np.log(max(0.03 * delay_percentile, 1e-5)), 0)

    def get_experience(self):
        return self.state_buf, self.action_buf, self.reward

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        data = {}
        data['payload'] = 'x' * 1400

        t = 0
        while True:
            t += 1
            state = self.get_curr_state()
            action = self.sample_action(state)

            for i in xrange(action):
                data['send_ts'] = curr_ts_ms()
                serialized_data = json.dumps(data)
                s.sendto(serialized_data, self.dest_addr)

            serialized_ack = s.recvfrom(1500)[0]
            ack = json.loads(serialized_ack)
            send_ts = ack['send_ts']
            ack_ts = ack['ack_ts']
            delay = ack_ts - send_ts
            self.delays.append(delay)

            if self.train:
                self.state_buf.append(state)
                self.action_buf.append(action)
                self.acked_bytes += ack['acked_bytes']
                self.first_ack_ts = min(ack_ts, self.first_ack_ts)
                self.last_ack_ts = max(ack_ts, self.last_ack_ts)

                if t >= self.max_steps:
                    break

        if self.train:
            self.duration = last_ack_ts - first_ack_ts
            self.compute_reward()

    def cleanup(self):
        self.s.close()
