#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan
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

import struct
import sys


class Message(object):
    total_size = 1396  # max usable size (header + payload)
    header_fmt = '>QQQQQ'
    header_size = struct.calcsize(header_fmt)
    payload_size = total_size - header_size
    dummy_payload = 'x' * payload_size

    def __init__(self, seq_num, send_ts, bytes_sent, ack_recv_ts, bytes_acked):
        # sequence number identifying each message
        self.seq_num = seq_num

        # timestamp (ms) of sending this message
        self.send_ts = send_ts

        # total bytes sent so far
        self.bytes_sent = bytes_sent

        # timestamp (ms) of receiving the last ack
        self.ack_recv_ts = ack_recv_ts

        # total bytes acked so far
        self.bytes_acked = bytes_acked

    # parse the message (only interested in header) received from the network
    @classmethod
    def parse(cls, message_string):
        if len(message_string) < Message.header_size:
            sys.exit('message is too small to contain header')

        return cls(*struct.unpack(Message.header_fmt,
                                  message_string[:Message.header_size]))

    # echo the message header directly without parsing
    @classmethod
    def transform_into_ack(cls, message_string):
        if len(message_string) < Message.header_size:
            sys.exit('message is too small to contain header')

        return message_string[:Message.header_size]

    # serialize the header to network byte order
    def header_to_string(self):
        return struct.pack(Message.header_fmt,
                           self.seq_num, self.send_ts, self.bytes_sent,
                           self.ack_recv_ts, self.bytes_acked)

    # serialize to a data message
    def to_data(self):
        return self.header_to_string() + Message.dummy_payload

    # serialize to an ack message (which simply echos the header)
    def to_ack(self):
        return self.header_to_string()
