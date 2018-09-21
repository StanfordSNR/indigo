#!/usr/bin/env python

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
import socket
import argparse

from message import Message


class Receiver(object):
    def __init__(self, port=0):
        # blocking UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))

        sys.stderr.write('[receiver] listening on port %s\n' %
                         self.sock.getsockname()[1])

    def cleanup(self):
        self.sock.close()

    def run(self):
        while True:
            msg_str, addr = self.sock.recvfrom(1500)

            data = Message.parse(msg_str)
            if data:
                self.sock.sendto(data.to_ack(), addr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int)
    args = parser.parse_args()

    try:
        receiver = Receiver(args.port)
        receiver.run()
    except KeyboardInterrupt:
        sys.stderr.write('Receiver is stopped\n')
    finally:
        receiver.cleanup()


if __name__ == '__main__':
    main()
