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
        sys.stderr.write('\nCleaning up...\n')
        self.sock.close()

    def construct_ack_from_data(self, serialized_data):
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
        TIMEOUT = 1000  # ms
        READ_ERR_FLAGS = h.READ_FLAGS | h.ERR_FLAGS

        retry_times = 0
        self.poller.modify(self.sock, READ_ERR_FLAGS)

        while True:
            self.sock.sendto('Hello from receiver', self.peer_addr)

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                retry_times += 1
                if retry_times > 3:
                    sys.stderr.write('Handshake failed after three retries\n')
                    return
                else:
                    sys.stderr.write('Handshake timed out and retrying...\n')
                    continue

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & h.ERR_FLAGS:
                    sys.exit('Channel closed or error occurred')

                if flag & h.READ_FLAGS:
                    msg, addr = self.sock.recvfrom(1500)

                    if addr != self.peer_addr:
                        continue

                    if msg != 'Hello from sender':
                        # 'Hello from sender' was presumably lost
                        # received subsequent data from peer sender
                        ack = self.construct_ack_from_data(msg)
                        if ack is not None:
                            self.sock.sendto(ack, self.peer_addr)

                    sys.stderr.write('Handshake success! '
                                     'Sender\'s address is %s:%s\n' % addr)
                    return

    def run(self):
        # handshake succeeded and receive data now
        while True:
            serialized_data, addr = self.sock.recvfrom(1500)

            if addr != self.peer_addr:
                continue

            ack = self.construct_ack_from_data(serialized_data)
            if ack is not None:
                self.sock.sendto(ack, self.peer_addr)
            else:
                continue
