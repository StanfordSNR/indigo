#!/usr/bin/env python

import sys
import argparse
from subprocess import check_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--username', default='francisyyan',
        help='username used for train cmd (default: francisyyan)')

    parser.add_argument(
        '--table', metavar='TABLE', nargs='?', default='TABLE',
        help='(messy) table of VM instances copied from Google Cloud Platform')
    args = parser.parse_args()

    cmd = 'grep -E -o "([0-9]{1,3}[\.]){3}[0-9]{1,3}" ' + args.table
    ip_list = check_output(cmd, shell=True).split()

    ret_cmd = ('~/RLCC/dagger/train.py --username %s --rlcc-dir '
               '/home/%s/RLCC --ps-hosts ') % (args.username, args.username)
    ret_int_ip_list = ''
    ret_ext_ip_list = ''

    worker_port = 16000
    for i in xrange(0, len(ip_list), 2):
        internal_ip = ip_list[i]
        external_ip = ip_list[i + 1]

        if i == 0:
            ret_cmd += internal_ip + ':15000 --worker-hosts '
        else:
            ret_cmd += internal_ip + ':%d,' % worker_port
            worker_port += 1

        ret_int_ip_list += '%s,' % internal_ip
        ret_ext_ip_list += '%s,' % external_ip

    print ret_cmd[:-1]
    print ret_int_ip_list[:-1]
    print ret_ext_ip_list[:-1]


if __name__ == '__main__':
    main()
