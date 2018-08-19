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
from os import path
from subprocess import check_call, Popen


def setup_local(args):
    if args.install_deps:
        cmd = 'sudo add-apt-repository -y ppa:keithw/mahimahi'
        sys.stderr.write('$ %s\n' % cmd)
        check_call(cmd, shell=True)

        cmd = 'sudo apt-get -y update'
        sys.stderr.write('$ %s\n' % cmd)
        check_call(cmd, shell=True)

        deps = 'mahimahi python-pip python-dev'
        cmd = 'sudo apt-get -yq --force-yes install %s' % deps
        sys.stderr.write('$ %s\n' % cmd)
        check_call(cmd, shell=True)

        cmd = 'sudo pip install --upgrade pip'
        sys.stderr.write('$ %s\n' % cmd)
        check_call(cmd, shell=True)

        cmd = 'sudo pip install numpy tensorflow'
        sys.stderr.write('$ %s\n' % cmd)
        check_call(cmd, shell=True)

    cmd = 'sudo sysctl -w net.ipv4.ip_forward=1'
    sys.stderr.write('$ %s\n' % cmd)
    check_call(cmd, shell=True)


def setup(args):
    if args.local:
        setup_local(args)
    else:
        procs = []
        ip_list = args.remote.split(',')

        for ip in ip_list:
            host = args.username + '@' + ip
            ssh_cmd = ['ssh', host, '-o', 'StrictHostKeyChecking=no']

            setup_src = path.join(args.rlcc_dir, 'helpers', 'setup.py')
            cmd = ssh_cmd + ['python', setup_src, '--local']

            if args.install_deps:
                cmd += ['--install-deps']

            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            procs.append(Popen(cmd))

        for proc in procs:
            proc.communicate()


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--local', action='store_true', help='setup on local machine')
    group.add_argument(
        '--remote', metavar='IP,...',
        help='comma-separated list of IP addresses of remote hosts')

    parser.add_argument(
        '--install-deps', action='store_true',
        help='install dependencies: tensorflow, mahimahi, etc.')
    parser.add_argument(
        '--username', default='francisyyan',
        help='username used in ssh connection (default: francisyyan)')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', default='/home/francisyyan/RLCC',
        help='absolute path to RLCC/ (default: /home/francisyyan/RLCC)')
    args = parser.parse_args()

    setup(args)


if __name__ == '__main__':
    main()
