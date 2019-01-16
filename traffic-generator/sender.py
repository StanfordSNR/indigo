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


import argparse
import ast
import bisect
import os
import socket
import subprocess
import sys
import time

import context  # noqa # pylint: disable=unused-import
import matplotlib.pyplot as plt
import numpy as np
from dagger.message import Message


class Function(object):
    def __init__(self, args, bin_step=0.01):
        self.addr = args.addr
        self.port = args.port
        self.dev = args.dev

        # validate points
        y = self.parse_points(args)
        x = list(np.arange(0, args.cycle, args.cycle/(len(y)-1.0)))
        x.append(float(args.cycle))
        assert len(x) == len(y)

        self.cycle = args.cycle
        self.lifetime = args.lifetime
        self.bin_step = bin_step
        self.num_bins = int(args.lifetime / bin_step)
        self.value_bins = []

        for bin_idx in xrange(self.num_bins):
            r = (bin_idx * bin_step) % self.cycle
            i = bisect.bisect(x, r)
            v_mbps = (y[i] - y[i-1]) * (r - x[i-1]) / (x[i] - x[i-1]) + y[i-1]

            v_bps = int(v_mbps * 1000 * 1000 / 8) * 8  # align to byte boundary
            self.value_bins.append(v_bps)

    def validate_points(self, points):
        # if points[0] != points[-1]:
        #     sys.exit('The first point must be same with the last point')

        if len(points) < 2:
            sys.exit('The number of points defining a function must be >= 2')

        for i in xrange(len(points)):
            if points[i] < 0:
                sys.exit('Invalid points: negative y-coordinate')

        return points

    def parse_points(self, args):
        points = []
        if args.sketch is not None:
            points = ast.literal_eval(args.sketch)
        elif args.trace is not None:
            with open(args.trace, 'rb') as fr:
                while True:
                    point = fr.readline()
                    if not point:
                        break
                    points.append(int(point))
        else:
            sys.exit('No valid points!')

        return points


def show_outline(func):
    x = list(np.arange(0, func.lifetime, func.lifetime / (len(func.value_bins)-1.0)))
    x.append(float(func.lifetime))
    y = func.value_bins

    plt.figure(figsize=(16, 6))
    plt.plot(x, y, 'b-', linewidth=1)
    plt.grid(True)

    plt.show()
    plt.close()


def generate_traffic(args):
    bind_addr = (args.addr, args.port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = 'x' * Message.total_size

    func = Function(args)

    # plot traffic outline
    # show_outline(func)

    pre_cmd_time = time.time()
    subprocess.Popen('tc qdisc add dev {} root fq maxrate {}'.format(
        func.dev, func.value_bins[0]), shell=True)
    cmd_time = time.time() - pre_cmd_time
    sleep_time = func.bin_step - cmd_time

    now = time.time()
    try:
        pid = os.fork()
        if pid == 0:  # child process
            while True:
                sender.sendto(msg, bind_addr)
        else:         # parent process
            for i in xrange(len(func.value_bins)):
                subprocess.Popen('tc qdisc change dev {} root fq maxrate {}'.format(
                    args.dev, func.value_bins[i]), shell=True)
                time.sleep(sleep_time)

            subprocess.call('kill {}'.format(pid), shell=True)
    except OSError:
        sys.exit(-1)

    exec_time = time.time() - now
    print('exec time: {}'.format(exec_time))
    sender.close()


def verify_args(args):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('addr')
    parser.add_argument('port', type=int)
    parser.add_argument('dev')
    parser.add_argument('-s', '--sketch', help='sketch of points to outline the shape of traffic')
    parser.add_argument('-c', '--cycle', type=float, help='cycle of sketch in seconds')
    parser.add_argument('-t', '--trace', help='trace of points to outline the shape of traffic')
    parser.add_argument('-l', '--lifetime', type=float, help='lifetime of traffic generator in seconds')
    args = parser.parse_args()

    # verify args
    verify_args(args)

    # prepare env
    subprocess.call('tc qdisc del dev {} root'.format(args.dev), shell=True)

    # start generator
    generate_traffic(args)


if __name__ == '__main__':
    main()
