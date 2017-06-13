#!/usr/bin/env python

import sys
import argparse
from subprocess import Popen


def build_cmd_db(args):
    cmd_db = {}

    cmd_db['git_clone'] = 'git clone https://github.com/StanfordSNR/RLCC.git'
    cmd_db['git_pull'] = ('cd %s && git reset --hard @ && git checkout master '
                          '&& git pull' % args.rlcc_dir)

    return cmd_db


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--remote', required=True, metavar='IP,...',
        help='comma-separated list of IP addresses of remote hosts')
    parser.add_argument(
        '--username', default='ubuntu',
        help='username used in ssh (default: ubuntu)')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', default='~/RLCC',
        help='path to RLCC/ (default: ~/RLCC)')
    args = parser.parse_args()
    cmd_db = build_cmd_db(args)

    ip_list = args.remote.split(',')
    procs = []

    for ip in ip_list:
        host = args.username + '@' + ip

        cmd = ['ssh', host, '-o', 'StrictHostKeyChecking=no',
               cmd_db['git_pull']]

        procs.append(Popen(cmd))

    for proc in procs:
        proc.communicate()


if __name__ == '__main__':
    main()
