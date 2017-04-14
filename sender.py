import sys
import json
import socket
import signal
import select
import numpy as np
import helpers as h
from helpers import curr_ts_ms


class Sender(object):
    def __init__(self, ip, port, training=False, debug=False):
        self.dest_addr = (ip, port)
        self.training = training
        self.debug = debug

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(0)

        self.poller = select.poll()
        self.poller.register(self.sock, h.ALL_FLAGS)

        # UDP datagram template
        self.data = {}
        self.data['payload'] = 'x' * 1400

        # dimension of state space and action space
        self.state_dim = 1
        self.action_cnt = 3

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 16.0

        # features in state vector
        self.base_delay = sys.maxint

        if self.training:
            self.running = True
            self.step_cnt = 0
            self.max_running_steps = 2000

            # statistics variables to compute rewards
            self.sent_bytes = 0
            self.acked_bytes = 0
            self.first_ack_ts = float('inf')
            self.last_ack_ts = 0
            self.total_delays = []

            # buffers for states and actions in a single episode
            self.state_buf = []
            self.action_buf = []

    def set_sample_action(self, sample_action):
        """Required to be called before running."""
        self.sample_action = sample_action

    def reset_training(self):
        assert self.training

        self.seq_num += 1
        self.next_ack = self.seq_num
        self.cwnd = 16.0

        self.base_delay = sys.maxint

        self.running = True
        self.step_cnt = 0

        self.sent_bytes = 0
        self.acked_bytes = 0
        self.first_ack_ts = float('inf')
        self.last_ack_ts = 0
        self.total_delays = []

        self.state_buf = []
        self.action_buf = []

        self.drain_packets()

    def drain_packets(self):
        """Drain all the packets left in the channel."""

        TIMEOUT = 1000  # ms
        READ_ERR_FLAGS = h.READ_FLAGS | h.ERR_FLAGS

        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                break

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & h.ERR_FLAGS:
                    self.sock.close()
                    sys.exit('Channel closed or error occurred')

                if flag & h.READ_FLAGS:
                    self.sock.recvfrom(1500)

    def update_state(self, ack):
        send_ts = ack['ack_send_ts']
        ack_ts = ack['ack_recv_ts']

        # queuing delay
        curr_delay = ack_ts - send_ts
        self.base_delay = min(self.base_delay, curr_delay)
        queuing_delay = curr_delay - self.base_delay

        if self.training:
            self.acked_bytes += ack['ack_bytes']
            self.total_delays.append(curr_delay)

            self.first_ack_ts = min(ack_ts, self.first_ack_ts)
            self.last_ack_ts = max(ack_ts, self.last_ack_ts)

            self.step_cnt += 1
            if self.step_cnt >= self.max_running_steps:
                self.running = False

        return [queuing_delay]

    def take_action(self, action):
        self.cwnd += action - 1

        if self.cwnd < 2.0:
            self.cwnd = 2.0

        if self.debug:
            sys.stderr.write('cwnd %.1f\n' % self.cwnd)

    def compute_reward(self):
        duration = self.last_ack_ts - self.first_ack_ts
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / duration
        delay_percentile = float(np.percentile(self.total_delays, 95))
        loss_rate = 1.0 - float(self.acked_bytes) / self.sent_bytes

        avg_throughput = max(0.0, min(12.0, avg_throughput))
        delay_percentile = max(20.0, min(220.0, delay_percentile))

        reward = 1.0
        reward += np.log(max(1e-5, avg_throughput / 12.0))
        reward += np.log(max(1e-5, (220.0 - delay_percentile) / 200.0))
        reward += np.log(max(1e-5, 1.0 - loss_rate))
        reward *= 10.0

        sys.stderr.write('Average throughput: %.2f Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile one-way delay: %d ms\n' %
                         delay_percentile)
        sys.stderr.write('Loss rate: %.2f\n' % loss_rate)
        sys.stderr.write('Reward: %.3f\n' % reward)

        return reward

    def get_experience(self):
        assert self.training

        reward = self.compute_reward()
        return self.state_buf, self.action_buf, reward

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        self.data['seq_num'] = self.seq_num
        self.seq_num += 1
        self.data['send_ts'] = curr_ts_ms()

        serialized_data = json.dumps(self.data)
        self.sock.sendto(serialized_data, self.dest_addr)

        if self.training:
            self.sent_bytes += len(serialized_data)

        if self.debug:
            sys.stderr.write('Sent seq_num %d\n' % self.data['seq_num'])

    def recv(self):
        serialized_ack = self.sock.recvfrom(1500)[0]
        ack = json.loads(serialized_ack)

        self.next_ack = max(self.next_ack, ack['ack_seq_num'] + 1)

        state = self.update_state(ack)
        action = self.sample_action(state)
        self.take_action(action)

        if self.training:
            self.state_buf.append(state)
            self.action_buf.append(action)

        if self.debug:
            sys.stderr.write('Received ack_seq_num %d\n' % ack['ack_seq_num'])

    def run(self):
        TIMEOUT = 500  # ms
        READ_ERR_FLAGS = h.READ_FLAGS | h.ERR_FLAGS

        self.poller.modify(self.sock, h.ALL_FLAGS)
        curr_flags = h.ALL_FLAGS

        while not self.training or self.running:
            if self.window_is_open():
                if curr_flags != h.ALL_FLAGS:
                    self.poller.modify(self.sock, h.ALL_FLAGS)
                    curr_flags = h.ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                self.send()

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & h.ERR_FLAGS:
                    self.sock.close()
                    sys.exit('Error occurred to the channel')

                if flag & h.READ_FLAGS:
                    self.recv()

                if flag & h.WRITE_FLAGS:
                    while self.window_is_open():
                        self.send()
