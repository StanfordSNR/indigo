#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import sys
import select
import socket
import argparse

from policy import Policy
from message import Message
import context
from helpers.utils import (READ_FLAGS, WRITE_FLAGS, ERR_FLAGS,
                           READ_ERR_FLAGS, ALL_FLAGS, timestamp_ms)


class Sender(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # non-blocking UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(0)  # set socket to non-blocking

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        # sequence numbers
        self.seq_num = 0
        self.next_ack = 0

        # congestion control policy
        self.policy = Policy()

    def cleanup(self):
        self.sock.close()

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.policy.cwnd

    def send(self):
        msg = Message(self.seq_num, timestamp_ms(), self.policy.bytes_sent,
                      self.policy.ack_recv_ts, self.policy.bytes_acked)
        self.sock.sendto(msg.to_data(), self.peer_addr)

        self.seq_num += 1

        # tell policy that a datagram was sent
        self.policy.data_sent(msg)

    def recv(self):
        msg_str, addr = self.sock.recvfrom(1500)
        ack = Message.parse(msg_str)

        # update next ACK's sequence number to expect
        self.next_ack = max(self.next_ack, ack.seq_num + 1)

        # tell policy that an ack was received
        self.policy.ack_received(ack)

    def run(self):
        while True:
            if self.window_is_open():
                self.poller.modify(self.sock, ALL_FLAGS)
            else:
                self.poller.modify(self.sock, READ_ERR_FLAGS)

            events = self.poller.poll(self.policy.timeout_ms())
            if not events:  # timed out; send one datagram to get rolling
                self.send()

            for fd, flag in events:
                if flag & ERR_FLAGS:
                    sys.exit('[sender] error returned from poller')

                if flag & READ_FLAGS:
                    self.recv()

                if flag & WRITE_FLAGS:
                    while self.window_is_open():
                        self.send()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    try:
        sender = Sender(args.ip, args.port)
        sender.run()
    except KeyboardInterrupt:
        sys.stderr.write('[sender] stopped\n')
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
