#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Wei Wang, Yiyang Shao (Huawei Technologies)
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

import argparse
import socket
import sys

from message import Message
from policy import Policy


class Receiver(object):
    def __init__(self, port=0):
        # blocking UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))

        sys.stderr.write('[receiver] listening on port %s\n' %
                         self.sock.getsockname()[1])

        self.delay_ack = True
        self.max_delay_ack_num = 2

# public
    def cleanup(self):
        self.sock.close()

    def run(self):
        delay_ack_count = 0
        while True:
            msg_str, addr = self.sock.recvfrom(1500)
            if msg_str == 'Hello':
                self.sock.sendto('Hello', addr)
                continue
            if Policy.delay_ack:
                delay_ack_count += 1
                if (delay_ack_count % Policy.delay_ack_count == 0):
                    self.sock.sendto(Message.transform_into_ack(msg_str), addr)
            else:
                self.sock.sendto(Message.transform_into_ack(msg_str), addr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    receiver = None
    try:
        receiver = Receiver(args.port)
        receiver.run()
    except KeyboardInterrupt:
        sys.stderr.write('[receiver] stopped\n')
    finally:
        if receiver:
            receiver.cleanup()


if __name__ == '__main__':
    main()
