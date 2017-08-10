import sys
import json
import socket
import select
from os import path
import numpy as np
import datagram_pb2
import project_root
from helpers.helpers import (
    curr_ts_ms, READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)


class Sender(object):
    def __init__(self, port=0):
        # UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        sys.stderr.write('[sender] Listening on port %s\n' %
                         self.sock.getsockname()[1])

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        # dummy data to fill in datagram
        self.send_ts = 1
        self.sent_bytes = 1
        self.delivered_time = 1
        self.delivered = 1
        self.dummy_payload = 'x' * 1400

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 0

    def cleanup(self):
        self.sock.close()

    def handshake(self):
        """Handshake with peer receiver. Must be called before run()."""

        while True:
            msg, addr = self.sock.recvfrom(1600)

            if msg == 'Hello from receiver' and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break

        self.sock.setblocking(0)  # non-blocking UDP socket

    def set_cwnd(self, cwnd):
        """Set constant cwnd. Must be called before run()."""

        self.cwnd = cwnd

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        data = datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = self.send_ts
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)

        self.seq_num += 1

    def recv(self):
        serialized_ack, addr = self.sock.recvfrom(1600)

        if addr != self.peer_addr:
            return

        ack = datagram_pb2.Ack()
        ack.ParseFromString(serialized_ack)

        self.next_ack = max(self.next_ack, ack.seq_num + 1)

    def run(self):
        TIMEOUT = 1000  # ms

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS

        while True:
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
