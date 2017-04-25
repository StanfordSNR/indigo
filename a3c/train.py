#!/usr/bin/env python

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
    # run parameter servers
    for i in xrange(len(args['ps_list'])):
        ssh_cmd = ['ssh', args['ps_list'][i]]

        cmd = ['python', args['worker_src'],
               '--ps-hosts', args['ps_hosts'],
               '--worker-hosts', args['worker_hosts'],
               '--job-name', 'ps',
               '--task-index', str(i)]
        cmd = ssh_cmd + cmd

        sys.stderr.write('$ %s\n' % ' '.join(cmd))
        args['ps_procs'].append(Popen(cmd, preexec_fn=os.setsid))

    # run workers
    port_list = []
    for i in xrange(len(args['worker_list'])):
        ssh_cmd = ['ssh', args['worker_list'][i]]

        # run worker to train sender
        port = str(get_open_udp_port())
        port_list.append(port)

        cmd = ['python', args['worker_src'],
               '--ps-hosts', args['ps_hosts'],
               '--worker-hosts', args['worker_hosts'],
               '--job-name', 'worker',
               '--task-index', str(i),
               '--port', port]
        cmd = ssh_cmd + cmd

        sys.stderr.write('$ %s\n' % ' '.join(cmd))
        args['worker_procs'].append(Popen(cmd, preexec_fn=os.setsid))

    # wait for senders to run
    time.sleep(3)

    # run receivers
    for i in xrange(len(args['worker_list'])):
        ssh_cmd = ['ssh', args['worker_list'][i]]

        mm_cmd = ['mm-delay', '20',
                  'mm-link', args['uplink_trace'], args['downlink_trace'],
                  '--downlink-queue=droptail',
                  '--downlink-queue-args=packets=200']
        port = port_list[i]
        recv_cmd = ['python', args['receiver_src'], '$MAHIMAHI_BASE', port]

        cmd = "%s -- sh -c '%s'" % (' '.join(mm_cmd), ' '.join(recv_cmd))
        cmd = ssh_cmd + [cmd]

        sys.stderr.write('$ %s\n' % ' '.join(cmd))
        args['receiver_procs'].append(Popen(cmd, preexec_fn=os.setsid))

    for ps_proc in args['ps_procs']:
        ps_proc.communicate()


def clean_up(args):
    sys.stderr.write('\nCleaning up...\n')

    hostname_list = args['worker_list'] + args['ps_list']
    for hostname in hostname_list:
        pkill_cmd = ('pkill -f %s; pkill -f mm-delay; pkill -f mm-link; '
                     'pkill -f mm-loss' % args['rlcc_dir'])
        kill_cmd = ['ssh', hostname, pkill_cmd]
        sys.stderr.write('$ %s\n' % ' '.join(kill_cmd))
        call(kill_cmd)

    # clean up
    procs = args['ps_procs'] + args['worker_procs'] + args['receiver_procs']
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError as e:
            sys.stderr.write('%s\n' % e)


def construct_args(prog_args):
    args = {}  # dictionary of arguments
    # file paths
    args['rlcc_dir'] = prog_args.rlcc_dir
    args['worker_src'] = path.join(args['rlcc_dir'], 'a3c', 'worker.py')
    args['receiver_src'] = path.join(
        args['rlcc_dir'], 'env', 'run_receiver.py')
    args['uplink_trace'] = path.join(args['rlcc_dir'], 'env', '12mbps.trace')
    args['downlink_trace'] = args['uplink_trace']

    # host names and processes
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
    args['receiver_procs'] = []

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
        '--username', required=True,
        help='username used in ssh connection')
    parser.add_argument(
        '--rlcc-dir', metavar='DIR', required=True,
        help='absolute path to RLCC/')
    prog_args = parser.parse_args()
    args = construct_args(prog_args)

    # run parameter servers and workers
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    finally:
        clean_up(args)


if __name__ == '__main__':
    main()
