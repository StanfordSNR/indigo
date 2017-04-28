import sys
import json
import socket
import select
import project_root
from helpers.helpers import (
    curr_ts_ms, READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, ALL_FLAGS)


class Receiver(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def cleanup(self):
        self.sock.close()

    def construct_ack_from_data(self, serialized_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        try:
            data = json.loads(serialized_data)
        except ValueError:
            return None

        ack = {}
        ack['ack_seq_num'] = data['seq_num']
        ack['ack_send_ts'] = data['send_ts']
        ack['ack_recv_ts'] = curr_ts_ms()
        ack['ack_bytes'] = len(serialized_data)
        return json.dumps(ack)

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
                if retry_times > 3:
                    sys.stderr.write(
                        '[receiver] Handshake failed after three retries\n')
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
        self.sock.setblocking(1)  # blocking UDP socket

        while True:
            serialized_data, addr = self.sock.recvfrom(1500)

            if addr == self.peer_addr:
                ack = self.construct_ack_from_data(serialized_data)
                if ack is not None:
                    self.sock.sendto(ack, self.peer_addr)
