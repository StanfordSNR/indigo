import sys
import json
import socket
import select
import numpy as np
import project_root
from helpers.helpers import (
    curr_ts_ms, READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)


class Sender(object):
    def __init__(self, port=0, train=False, debug=False):
        self.train = train
        self.debug = debug

        # UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        sys.stderr.write('[sender] Listening on port %s\n' %
                         self.sock.getsockname()[1])

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        # UDP datagram template
        self.data = {}
        self.data['payload'] = 'x' * 1400

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10.0

        # dimension of state space and action space
        self.state_dim = 3
        self.action_cnt = 5
        self.action_mapping = [-0.5, -0.25, 0.0, 0.25, 0.5]

        # features (in state vector) related
        self.base_delay = sys.maxint
        self.send_ewma = 0.0
        self.ack_ewma = 0.0
        self.prev_send_ts = None
        self.prev_ack_ts = None

        if self.train:
            self.running = True
            self.step_cnt = 0
            self.episode_len = 2000

            # statistics variables to compute rewards
            self.sent_bytes = 0
            self.acked_bytes = 0
            self.first_ack_ts = float('inf')
            self.last_ack_ts = 0
            self.total_delays = []

            # buffers for states and actions in a single episode
            self.state_buf = []
            self.action_buf = []

    def cleanup(self):
        self.sock.close()

    def handshake(self):
        """Handshake with peer receiver. Must be called before run()."""

        while True:
            msg, addr = self.sock.recvfrom(1500)

            if msg == 'Hello from receiver' and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender]: Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break

        self.sock.setblocking(0)  # non-blocking UDP socket

    def set_sample_action(self, sample_action):
        """Set the policy. Must be called before run()."""

        self.sample_action = sample_action

    def reset(self):
        """Reset the sender. Must be called in every training iteration."""

        assert self.train

        self.seq_num += 1
        self.next_ack = self.seq_num
        self.cwnd = 10.0

        self.base_delay = sys.maxint
        self.send_ewma = 0.0
        self.ack_ewma = 0.0
        self.prev_send_ts = None
        self.prev_ack_ts = None

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
        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                break

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    self.sock.recvfrom(1500)

    def update_state(self, ack):
        send_ts = ack['ack_send_ts']
        ack_ts = ack['ack_recv_ts']

        # queuing delay
        curr_delay = ack_ts - send_ts
        self.base_delay = min(self.base_delay, curr_delay)
        queuing_delay = curr_delay - self.base_delay

        # EWMA of interarrival time between two send_ts and two ack_ts
        if self.prev_send_ts is None:
            assert self.prev_ack_ts is None
            self.prev_send_ts = send_ts
            self.prev_ack_ts = ack_ts

        send_ts_diff = send_ts - self.prev_send_ts
        ack_ts_diff = ack_ts - self.prev_ack_ts
        self.prev_send_ts = send_ts
        self.prev_ack_ts = ack_ts

        alpha = 0.125
        self.send_ewma = (1 - alpha) * self.send_ewma + alpha * send_ts_diff
        self.ack_ewma = (1 - alpha) * self.ack_ewma + alpha * ack_ts_diff

        if self.train:
            self.acked_bytes += ack['ack_bytes']
            self.total_delays.append(curr_delay)

            self.first_ack_ts = min(ack_ts, self.first_ack_ts)
            self.last_ack_ts = max(ack_ts, self.last_ack_ts)

            self.step_cnt += 1
            if self.step_cnt >= self.episode_len:
                self.running = False

        return [queuing_delay, self.send_ewma, self.ack_ewma]

    def take_action(self, action):
        self.cwnd += self.action_mapping[action]

        if self.cwnd < 5.0:
            self.cwnd = 5.0

        if self.debug:
            sys.stderr.write('cwnd %.2f\n' % self.cwnd)

    def compute_reward(self):
        duration = self.last_ack_ts - self.first_ack_ts
        avg_throughput = float(self.acked_bytes * 8) * 0.001 / duration
        delay_percentile = float(np.percentile(self.total_delays, 95))
        loss_rate = 1.0 - float(self.acked_bytes) / self.sent_bytes

        avg_throughput = max(0.0, min(12.0, avg_throughput))
        delay_percentile = max(20.0, min(230.0, delay_percentile))

        reward = 3.0
        reward += np.log(max(1e-5, avg_throughput / 12.0))
        reward += np.log(max(1e-5, (230.0 - delay_percentile) / 210.0))
        reward += np.log(max(1e-5, 1.0 - loss_rate))
        reward *= 10.0

        sys.stderr.write('Average throughput: %.2f Mbps\n' % avg_throughput)
        sys.stderr.write('95th percentile one-way delay: %d ms\n' %
                         delay_percentile)
        sys.stderr.write('Loss rate: %.2f\n' % loss_rate)
        sys.stderr.write('Reward: %.3f\n' % reward)

        return reward

    def get_experience(self):
        assert self.train

        reward = self.compute_reward()
        return self.state_buf, self.action_buf, reward

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        self.data['seq_num'] = self.seq_num
        self.seq_num += 1
        self.data['send_ts'] = curr_ts_ms()

        serialized_data = json.dumps(self.data)
        self.sock.sendto(serialized_data, self.peer_addr)

        if self.train:
            self.sent_bytes += len(serialized_data)

        if self.debug:
            sys.stderr.write('Sent seq_num %d\n' % self.data['seq_num'])

    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1500)

        if addr != self.peer_addr:
            return

        try:
            ack = json.loads(serialized_ack)
        except ValueError:
            return

        self.next_ack = max(self.next_ack, ack['ack_seq_num'] + 1)

        state = self.update_state(ack)
        action = self.sample_action(state)
        self.take_action(action)

        if self.train:
            self.state_buf.append(state)
            self.action_buf.append(action)

        if self.debug:
            sys.stderr.write('Received ack_seq_num %d\n' % ack['ack_seq_num'])

    def run(self):
        TIMEOUT = 500  # ms

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS

        while not self.train or self.running:
            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                self.send()

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    self.recv()

                if flag & WRITE_FLAGS:
                    while self.window_is_open():
                        self.send()
