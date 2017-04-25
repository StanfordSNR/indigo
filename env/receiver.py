import sys
import json
import socket
import select
import project_root
import helpers.helpers as h
from helpers.helpers import curr_ts_ms


class Receiver(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.poller = select.poll()
        self.poller.register(self.sock, h.ALL_FLAGS)

        # handshake with peer sender
        self.sock.setblocking(0)
        self.handshake()
        self.sock.setblocking(1)

    def clean_up(self):
        self.sock.close()
        sys.stderr.write('\nCleaned up\n')

    def handshake(self):
        TIMEOUT = 1000  # ms
        retry_times = 0
        self.poller.modify(self.sock, h.READ_FLAGS)

        while True:
            self.sock.sendto('Hello from receiver', self.peer_addr)

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times >= 3:
                    return
                else:
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & h.READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1500)

                    if msg == 'Hello from sender' and addr == self.peer_addr:
                        return

    def run(self):
        ack = {}
        while True:
            serialized_data, addr = self.sock.recvfrom(1500)

            if addr != self.peer_addr:
                continue

            try:
                data = json.loads(serialized_data)
            except ValueError:
                continue

            ack['ack_seq_num'] = data['seq_num']
            ack['ack_send_ts'] = data['send_ts']
            ack['ack_recv_ts'] = curr_ts_ms()
            ack['ack_bytes'] = len(serialized_data)
            self.sock.sendto(json.dumps(ack), self.peer_addr)
