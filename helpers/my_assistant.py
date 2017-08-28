#!/usr/bin/env python

import sys
import argparse
from subprocess import Popen, check_call, check_output


def run_cmd(args, host, procs):
    cmd = args.cmd
    cmd_in_ssh = None

    if cmd == 'copy_key':
        cmd_to_run = ('KEY=$(cat ~/.ssh/id_rsa.pub); '
                      'ssh -o StrictHostKeyChecking=no %s '
                      '"grep -qF \'$KEY\' .ssh/authorized_keys || '
                      'echo \'$KEY\' >> .ssh/authorized_keys"' % host)
        check_call(cmd_to_run, shell=True)

    elif cmd == 'git_clone':
        cmd_in_ssh = 'git clone https://github.com/StanfordSNR/RLCC.git'

    elif cmd == 'git_pull':
        cmd_in_ssh = ('cd %s && git checkout fyy-lstm-50 && '
                      'git reset --hard @~1 && git pull' % args.rlcc_dir)

    elif cmd == 'rm_history':
        cmd_in_ssh = ('rm -f %s/history' % args.rlcc_dir)

    elif cmd == 'cp_history':
        cmd_to_run = ('rsync --ignore-missing-args %s:%s/history %s/%s_history'
                      % (host, args.rlcc_dir, args.local_rlcc_dir, host))
        check_call(cmd_to_run, shell=True)

    else:
        cmd_in_ssh = cmd

    if cmd_in_ssh:
        cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', host, cmd_in_ssh]
        procs.append(Popen(cmd))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--remote', required=True, metavar='IP,...',
        help='comma-separated list of IP addresses of remote hosts')
    parser.add_argument(
        '--username', default='francisyyan',
        help='username used in ssh (default: francisyyan)')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', default='~/RLCC',
        help='path to RLCC/ (default: ~/RLCC)')
    parser.add_argument(
        '--local-rlcc-dir', metavar='DIR', default='~/RLCC',
        help='path to RLCC/ (default: ~/RLCC)')
    parser.add_argument('cmd')
    args = parser.parse_args()

    ip_list = args.remote.split(',')
    procs = []

    for ip in ip_list:
        host = args.username + '@' + ip
        run_cmd(args, host, procs)

    for proc in procs:
        proc.communicate()


if __name__ == '__main__':
    main()
