# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Huawei Technologies
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
# import json
import socket
import select
import struct
import signal
from os import path
# import datagram_pb2
import project_root
from helpers.helpers import READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, ALL_FLAGS

received_bytes = 0
test_name_str = None


def on_quit(a, b):
    if test_name_str is not None:
        file = path.join(project_root.DIR, 'tests', 'rtt_loss', 'receiver_'+test_name_str)
        file = open(file, 'w')
        file.write(str(received_bytes)+'\n')
        file.close()


class Receiver(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        signal.signal(signal.SIGTERM, on_quit)
        signal.signal(signal.SIGQUIT, on_quit)
        signal.signal(signal.SIGALRM, on_quit)

    def set_test_name(self, name):
        global test_name_str
        test_name_str = name

    def cleanup(self):
        self.sock.close()

    # def construct_ack_from_data(self, serialized_data):
    def construct_ack_from_data(self, unpacked_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        # data = datagram_pb2.Data()
        # data.ParseFromString(serialized_data)

        # ack = datagram_pb2.Ack()
        # ack.seq_num = data.seq_num
        # ack.send_ts = data.send_ts
        # ack.sent_bytes = data.sent_bytes
        # ack.delivered_time = data.delivered_time
        # ack.delivered = data.delivered
        # ack.ack_bytes = len(serialized_data)
        if len(unpacked_data) < 28:
            return None

        return struct.pack('!28si', unpacked_data[:28], len(unpacked_data) * 2)

        # return ack.SerializeToString()

    def handshake(self):
        """Handshake with peer sender. Must be called before run()."""

        self.sock.setblocking(0)  # non-blocking UDP socket

        TIMEOUT = 1000  # ms

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            self.sock.sendto('Hello from receiver', self.peer_addr)
            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times > 10:
                    sys.stderr.write(
                        '[receiver] Handshake failed after 10 retries\n')
                    return
                else:
                    sys.stderr.write(
                        '[receiver] Handshake timed out and retrying...\n')
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1500)

                    if addr == self.peer_addr:
                        if msg != 'Hello from sender':
                            # 'Hello from sender' was presumably lost
                            # received subsequent data from peer sender
                            ack = self.construct_ack_from_data(msg)
                            if ack is not None:
                                self.sock.sendto(ack, self.peer_addr)
                        return

    def run(self):
        global received_bytes
        self.sock.setblocking(1)  # blocking UDP socket

        on_ack = 0
        try:
            while True:
                serialized_data, addr = self.sock.recvfrom(1500)
                received_bytes = received_bytes + len(serialized_data)
                on_ack += 1
                if on_ack == 2:
                    on_ack = 0
                    if addr == self.peer_addr:
                        ack = self.construct_ack_from_data(serialized_data)
                        if ack is not None:
                            self.sock.sendto(ack, self.peer_addr)
        except KeyboardInterrupt:
            on_quit(0,0)
        finally:
            self.sock.close()
