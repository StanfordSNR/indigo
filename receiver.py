#!/usr/bin/env python

import sys
import json
import socket
from helpers import curr_ts_ms


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', 10000))
    ip, port = s.getsockname()
    sys.stderr.write('Listening on port: %s\n' % port)

    ack = {}
    while True:
        raw_data, addr = s.recvfrom(1500)
        data = json.loads(raw_data)

        ack['send_ts'] = data['send_ts']
        s.sendto(json.dumps(ack), addr)


if __name__ == '__main__':
    main()
