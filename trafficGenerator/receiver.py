#!/usr/bin/env python

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


import socket
import sys


def main(args):
    bind_addr = (args[1], int(args[2]))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(bind_addr)

    while True:
        data, addr = s.recvfrom(1500)
        if not data:
            print('ERROR: null msg from client.')
            break
        # print('receive {} from {}'.format(len(data), addr))
    s.close()


if __name__ == '__main__':
    main(sys.argv)
