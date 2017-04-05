import sys
import json
import socket
from helpers import curr_ts_ms


class Receiver(object):
    def __init__(self, port=0):
        self.port = port

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s = self.s
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', self.port))
        sys.stderr.write('Listening on port: %s\n' % s.getsockname()[1])

        ack = {}
        while True:
            serialized_data, addr = s.recvfrom(1500)
            data = json.loads(serialized_data)
            ack['send_ts'] = data['send_ts']
            ack['ack_ts'] = curr_ts_ms()
            ack['acked_bytes'] = len(serialized_data)
            s.sendto(json.dumps(ack), addr)

    def cleanup(self):
        self.s.close()
