#!/usr/bin/env python

import json
import socket
import argparse
from helpers import curr_ts_ms
from controller import get_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    data = {}
    data['payload'] = 'x' * 1400
    rtts = []
    while True:
        # determine action
        send_cnt = get_action(rtts)
        for i in xrange(send_cnt):
            data['send_ts'] = curr_ts_ms()
            s.sendto(json.dumps(data), (args.ip, args.port))

        # wait for ACK
        raw_ack, _ = s.recvfrom(1500)
        ack = json.loads(raw_ack)
        rtt = curr_ts_ms() - ack['send_ts']
        rtts.append(rtt)


if __name__ == '__main__':
    main()
