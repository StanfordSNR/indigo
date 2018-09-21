from message import Message

import context
from helpers.utils import curr_ts_ms


class Policy(object):
    min_cwnd = 10
    max_cwnd = 18000

    def __init__(self):
        self.cwnd = 10

        self.bytes_sent = 0
        self.ack_recv_ts = 0
        self.bytes_acked = 0

    def timeout_ms(self):
        return -1

    def ack_received(self, ack):
        self.ack_recv_ts = curr_ts_ms()
        self.bytes_acked += Message.total_size

        # TODO: test only; slow start until reach max cwnd
        self.cwnd = max(Policy.min_cwnd, min(Policy.max_cwnd, self.cwnd + 1))

    def data_sent(self, data):
        self.bytes_sent += Message.total_size
