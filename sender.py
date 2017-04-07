import sys
import json
import socket
import signal
import numpy as np
from helpers import curr_ts_ms, RingBuffer, TimeoutError, timeout_handler


class Sender(object):
    def __init__(self, ip, port):
        self.dest_addr = (ip, port)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.data = {}
        self.data['payload'] = 'x' * 1400

    # required to be called before running
    def setup(self, **params):
        self.training = params['training']
        self.state_dim = params['state_dim']
        self.sample_action = params['sample_action']

        self.delay_buf = RingBuffer(self.state_dim)

        if self.training:
            self.setup_training(params)

    def setup_training(self, params):
        self.delay_weight = params['delay_weight']
        self.loss_weight = params['loss_weight']

        self.state_buf = []
        self.action_buf = []

        # for reward computation
        self.sent_bytes = 0
        self.acked_bytes = 0
        self.total_delays = []

    def reset(self):
        self.delay_buf.reset()
        self.state_buf = []
        self.action_buf = []
        self.sent_bytes = 0
        self.acked_bytes = 0
        self.total_delays = []

        # workaround to drain packets left in the queue
        signal.signal(signal.SIGALRM, timeout_handler)
        while True:
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            try:
                self.s.recvfrom(1500)
            except TimeoutError:
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

    def get_curr_state(self):
        return self.delay_buf.get()

    def compute_reward(self):
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / self.duration
        delay_percentile = float(np.percentile(self.total_delays, 95))
        loss_rate = 1.0 - float(self.acked_bytes) / self.sent_bytes

        sys.stderr.write('Average throughput: %.2f Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile one-way delay: %d ms\n' %
                         delay_percentile)
        sys.stderr.write('Loss rate: %.2f\n' % loss_rate)

        self.reward = np.log(max(avg_throughput, 1e-5))
        self.reward -= self.delay_weight * max(
                       np.log(max(delay_percentile, 1e-5)), 0)
        self.reward += self.loss_weight * np.log(1.0 - loss_rate)

    def get_experience(self):
        self.compute_reward()
        return self.state_buf, self.action_buf, self.reward

    def send(self, times):
        for i in xrange(times):
            self.data['send_ts'] = curr_ts_ms()
            serialized_data = json.dumps(self.data)
            self.s.sendto(serialized_data, self.dest_addr)

            if self.training:
                self.sent_bytes += len(serialized_data)

    def recv(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        while True:
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            try:
                serialized_ack = self.s.recvfrom(1500)[0]
            except TimeoutError:
                self.send(1)
            else:
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)

        return json.loads(serialized_ack)

    def run(self):
        if self.training:
            first_ack_ts = sys.maxint
            last_ack_ts = 0

        self.send(2)
        while True:
            state = self.get_curr_state()
            action = self.sample_action(state)
            self.send(action)
            ack = self.recv()

            send_ts = ack['send_ts']
            ack_ts = ack['ack_ts']
            delay = ack_ts - send_ts
            self.delay_buf.append(delay)

            if self.training:
                self.state_buf.append(state)
                self.action_buf.append(action)

                self.acked_bytes += ack['acked_bytes']
                first_ack_ts = min(ack_ts, first_ack_ts)
                last_ack_ts = max(ack_ts, last_ack_ts)
                self.total_delays.append(delay)

                if last_ack_ts - first_ack_ts > 5000:
                    break

        if self.training:
            self.duration = last_ack_ts - first_ack_ts

    def cleanup(self):
        self.s.close()
