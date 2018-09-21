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


import sys
import argparse
from subprocess_wrappers import Popen
from helpers import ssh_cmd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ip', required=True, metavar='IP,...',
        help='comma-separated list of IP addresses of remote hosts')
    parser.add_argument(
        '--user', default='ubuntu',
        help='username used in ssh (default: ubuntu)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ssh', metavar='CMD', help='commands to run over SSH')
    group.add_argument('--cmd', metavar='CMD', help='predefined commands')
    args = parser.parse_args()

    ip_list = args.ip.split(',')
    procs = []

    sys.stderr.write('%d IPs in total\n' % len(ip_list))

    for ip in ip_list:
        host = args.user + '@' + ip

        if args.ssh is not None:
            # run commands over SSH
            procs.append(Popen(ssh_cmd(host) + [args.ssh]))
        elif args.cmd is not None:
            if args.cmd == 'remove_key':
                procs.append(Popen(['ssh-keygen', '-f',
                    '/home/%s/.ssh/known_hosts' % args.user, '-R', ip]))

    for proc in procs:
        proc.communicate()


if __name__ == '__main__':
    main()
