#!/usr/bin/env python

import sys
import socket


class PerfClient(object):

    def __init__(self):
        self.socket = None

        self.rtts = []

    def cleanup(self):
        if self.socket:
            self.socket.close()

    def connect_perf_server(self, port):
        self.address = ('0.0.0.0', port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect(self.address)
        except socket.error:
            sys.stderr.write('connect to perf server error\n')
            return -1
        return 0

    def collect_perf_data(self, policy):

        # collect data from policy
        self.rtts.append(policy.rtt_ewma)

        # send perf data to perf client
        # self.socket.send(perf_data)
