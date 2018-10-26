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
import socket
import subprocess
import sys
import time
import argparse
import ast
import bisect


class Function(object):
    def __init__(self, points, step_s=0.01):
        # validate points
        points = sorted(points, key=lambda tup: tup[0])
        assert(points[0][0] == 0)
        assert(points[-1][0] > 0)
        self.period = points[-1][0]

        x = []
        y = []
        for point in points:
            x.append(point[0])
            y.append(point[1])

        self.step_s = step_s
        self.value_bins = []

        # calculate value_bins
        lower_s = 0.0
        upper_s = 60.0

        curr_s = lower_s
        while curr_s < upper_s:
            residue = curr_s % self.period
            i = bisect.bisect(x, residue)
            f = (y[i] - y[i-1]) * (residue - x[i-1]) / (x[i] - x[i-1]) + y[i-1]

            value_bps = int(f * 1000 * 1000 / 8) * 8
            self.value_bins.append(value_bps)
            curr_s += self.step_s


def generate_traffic(addr, port, dev, points):
    bind_addr = (addr, port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = 'x' * 1392
    func = Function(points)

    pre_cmd_time = time.time()
    subprocess.Popen('tc qdisc add dev {} root fq maxrate {}'.format(
                     dev, func.value_bins[0]), shell=True)
    cmd_time = time.time() - pre_cmd_time
    sleep_time = func.step_s - cmd_time

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
