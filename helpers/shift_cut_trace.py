#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_trace', metavar='INPUT-TRACE',
        help='input trace file that needs to shift to start from 0 '
        'and cut into 60 seconds long')
    parser.add_argument(
        'output_trace', metavar='OUTPUT-TRACE',
        help='output trace file after shifting and cutting')
    args = parser.parse_args()

    input_trace = open(args.input_trace)
    output_trace = open(args.output_trace, 'w')

    starter_ts = None
    while True:
        line = input_trace.readline()
        if not line:
            break

        ts = int(line)
        if ts < 10000:
            continue

        if starter_ts is None:
            starter_ts = ts

        if ts <= 70000:
            output_trace.write('%d\n' % (ts - starter_ts))
        else:
            break

    input_trace.close()
    output_trace.close()


if __name__ == '__main__':
    main()
