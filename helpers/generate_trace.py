#!/usr/bin/env python

import argparse
import numpy as np
from os import path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bandwidth', metavar='Mbps', required=True, type=int,
                        help='integer constant bandwidth (Mbps)')
    parser.add_argument('--output-dir', metavar='DIR', required=True,
                        help='directory to output trace')
    args = parser.parse_args()

    # trace path
    trace_path = path.join(args.output_dir, '%dmbps.trace' % args.bandwidth)
    trace = open(trace_path, 'w')

    # number of packets per 12 ms
    pkts_rate = args.bandwidth

    # each bucket consists of 12 ms, so the trace is 60 seconds long
    for bucket in xrange(0, 5000):
        ts_list = np.random.randint(bucket * 12, (bucket + 1) * 12, pkts_rate)
        ts_list = np.sort(ts_list)

        for ts in ts_list:
            trace.write('%s\n' % ts)

    trace.close()


if __name__ == '__main__':
    main()
