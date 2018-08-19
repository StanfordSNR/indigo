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


import os
import sys
import time
import signal
import argparse
import project_root
from os import path
from subprocess import Popen, call
from helpers.helpers import get_open_udp_port


def run(args):
    # run worker.py on ps and worker hosts
    for job_name in ['ps', 'worker']:
        host_list = args[job_name + '_list']
        procs = args[job_name + '_procs']

        for i in xrange(len(host_list)):
            ssh_cmd = ['ssh', host_list[i]]

            cmd = ['python', args['worker_src'],
                   '--ps-hosts', args['ps_hosts'],
                   '--worker-hosts', args['worker_hosts'],
                   '--job-name', job_name,
                   '--task-index', str(i)]

            cmd = ssh_cmd + cmd

            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            procs.append(Popen(cmd, preexec_fn=os.setsid))

    # ps will block forever
    for ps_proc in args['ps_procs']:
        ps_proc.communicate()


def cleanup(args):
    all_procs = args['ps_procs'] + args['worker_procs']
    for proc in all_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError as e:
            sys.stderr.write('%s\n' % e)

    host_set = set(args['ps_list'] + args['worker_list'])
    pkill_script = path.join(args['rlcc_dir'], 'helpers', 'pkill.py')

    for host in host_set:
        kill_cmd = ['ssh', host, 'python', pkill_script, args['rlcc_dir']]
        sys.stderr.write('$ %s\n' % ' '.join(kill_cmd))
        call(kill_cmd)

    sys.stderr.write('\nAll cleaned up.\n')


def construct_args(prog_args):
    # construct a dictionary of arguments
    args = {}

    # file paths
    args['rlcc_dir'] = prog_args.rlcc_dir
    args['worker_src'] = path.join(args['rlcc_dir'], 'dagger', 'worker.py')

    # hostnames and processes
    args['ps_hosts'] = prog_args.ps_hosts
    args['worker_hosts'] = prog_args.worker_hosts

    args['ps_list'] = prog_args.ps_hosts.split(',')
    args['worker_list'] = prog_args.worker_hosts.split(',')
    args['username'] = prog_args.username

    for i, host in enumerate(args['ps_list']):
        args['ps_list'][i] = args['username'] + '@' + host.split(':')[0]

    for i, host in enumerate(args['worker_list']):
        args['worker_list'][i] = args['username'] + '@' + host.split(':')[0]

    args['ps_procs'] = []
    args['worker_procs'] = []

    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ps-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument(
        '--worker-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of workers')
    parser.add_argument(
        '--username', default='ubuntu',
        help='username used in ssh connection (default: ubuntu)')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', default='/home/ubuntu/RLCC',
        help='absolute path to RLCC/ (default: /home/ubuntu/RLCC)')
    prog_args = parser.parse_args()
    args = construct_args(prog_args)

    # run worker.py on ps and worker hosts
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(args)


if __name__ == '__main__':
    main()
