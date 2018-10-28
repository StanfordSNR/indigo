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


import os
import sys
import socket
import subprocess
import time
import ast
import bisect
import argparse


class Function(object):
    def __init__(self, points, bin_s=0.01, max_s=60.0):
        # sort and validate points
        points = sorted(points, key=lambda tup: tup[0])
        self.validate_points(points)
        self.points = points

        # the last point defines the period of the function
        self.period = points[-1][0]

        # unzip points
        x, y = zip(*points)

        # function's value in each bin
        self.bin_s = bin_s
        self.max_s = max_s
        num_bins = int(max_s / bin_s)
        self.value_bins = []

        for bin_idx in xrange(num_bins):
            r = (bin_idx * bin_s) % self.period
            i = bisect.bisect(x, r)
            v_mbps = (y[i] - y[i-1]) * (r - x[i-1]) / (x[i] - x[i-1]) + y[i-1]

            v_bps = int(v_mbps * 1000 * 1000 / 8) * 8  # align to byte boundary
            self.value_bins.append(v_bps)

    def validate_points(self, points):
        if points[0][0] != 0:
            sys.exit('The first point must define at x=0')

        if len(points) < 2:
            sys.exit('The number of points defining a function must be >= 2')

        for i in xrange(len(points)):
            if points[i][1] < 0:
                sys.exit('Invalid points: negative y-coordinate')

            if i > 0 and points[i][0] == points[i - 1][0]:
                sys.exit('Invalid points: same x-coordinate')



def generate_traffic(addr, port, dev, points):
    bind_addr = (addr, port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = 'x' * 1392
    func = Function(points)

    pre_cmd_time = time.time()
    subprocess.Popen('tc qdisc add dev {} root fq maxrate {}'.format(
                     dev, func.value_bins[0]), shell=True)
    cmd_time = time.time() - pre_cmd_time
    sleep_time = func.bin_s - cmd_time

    now = time.time()
    try:
        pid = os.fork()
        if pid == 0:  # child process
            while True:
                sender.sendto(msg, bind_addr)
        else:         # parent process
            for i in xrange(len(func.value_bins)):
                subprocess.Popen(
                    'tc qdisc change dev {} root fq maxrate {}'.format(
                    dev, func.value_bins[i]), shell=True)
                time.sleep(sleep_time)

            subprocess.call('kill {}'.format(pid), shell=True)
    except OSError:
        sys.exit(-1)

    exec_time = time.time() - now
    print('exec time: {}'.format(exec_time))
    sender.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('addr')
    parser.add_argument('port', type=int)
    parser.add_argument('dev')
    parser.add_argument('points')
    args = parser.parse_args()

    f = Function(ast.literal_eval(args.points))

    # prepare env
    subprocess.call('tc qdisc del dev {} root'.format(args.dev), shell=True)

    # start to generate
    generate_traffic(args.addr, args.port, args.dev,
                     ast.literal_eval(args.points))


if __name__ == '__main__':
    main()
