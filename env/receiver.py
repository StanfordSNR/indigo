import sys
import json
import socket
import project_root
from helpers.helpers import curr_ts_ms


class Receiver(object):
    def __init__(self, port=0):
        self.port = port

    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('0.0.0.0', self.port))
            sys.stderr.write('Listening on port: %s\n' % sock.getsockname()[1])

            ack = {}
            while True:
                serialized_data, addr = sock.recvfrom(1500)
                data = json.loads(serialized_data)

                ack['ack_seq_num'] = data['seq_num']
                ack['ack_send_ts'] = data['send_ts']
                ack['ack_recv_ts'] = curr_ts_ms()
                ack['ack_bytes'] = len(serialized_data)
                sock.sendto(json.dumps(ack), addr)
        finally:
            sock.close()
