#!/usr/bin/env python

import os
import sys
import signal
import argparse
import project_root
from os import path
from subprocess import Popen, call
from helpers.helpers import get_open_udp_port


def retrieve_hostnames(args):
    ps_hostname = args.username + '@' + args.ps.split(':')[0]

    workers = args.workers.split(',')
    worker_hostnames = []
    for worker in workers:
        worker_hostname = args.username + '@' + worker.split(':')[0]
        worker_hostnames.append(worker_hostname)

    return [ps_hostname] + worker_hostnames


def run(args):
    hostnames = retrieve_hostnames(args)

    worker_src = path.join(args.rlcc_dir, 'a3c', 'worker.py')
    receiver_src = path.join(args.rlcc_dir, 'env', 'run_receiver.py')
    uplink_trace = path.join(args.rlcc_dir, 'env', '12mbps.trace')
    downlink_trace = uplink_trace

    procs = [None] * len(hostnames)
    for i in xrange(len(hostnames)):
        ssh_cmd = ['ssh', hostnames[i]]

        if i == 0:
            cmd = ssh_cmd
            cmd += ['python', worker_src,
                    '--ps', args.ps, '--workers', args.workers,
                    '--job-name', 'ps', '--task-index', '0']
            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            procs[i] = Popen(cmd, preexec_fn=os.setsid)
        else:
            # run receiver
            cmd = ssh_cmd + ['python', receiver_src, str(get_open_udp_port())]
            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            Popen(cmd)

            # run worker that trains sender
            cmd = ssh_cmd
            cmd += ['mm-delay', '20',
                    'mm-link', uplink_trace, downlink_trace,
                    '--uplink-queue=droptail',
                    '--uplink-queue-args=packets=200',
                    '--', 'sh', '-c']
            cmd += ['python', worker_src,
                    '--ps', args.ps, '--workers', args.workers,
                    '--job-name', 'worker', '--task-index', str(i - 1)]
            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            procs[i] = Popen(cmd, preexec_fn=os.setsid)

    try:
        procs[0].communicate()
    except:
        pass
    finally:
        sys.stderr.write('\nCleaning up...\n')

        for hostname in hostnames:
            pkill_cmd = ('pkill -f %s; pkill -f mm-delay; pkill -f mm-link; '
                         'pkill -f mm-loss' % args.rlcc_dir)
            kill_cmd = ['ssh', hostname, pkill_cmd]
            sys.stderr.write('$ %s\n' % ' '.join(kill_cmd))
            call(kill_cmd)

        # clean up
        for proc in procs:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps', required=True,
                        help='IP:port of parameter server')
    parser.add_argument('--workers', required=True,
                        help='comma-separated list of IP:port of workers')
    parser.add_argument('--username', default='ubuntu',
                        help='username ahead of IP')
    parser.add_argument('--rlcc-dir', default='/home/ubuntu/RLCC/',
                        help='absolute path to RLCC/')
    args = parser.parse_args()

    # run workers
    run(args)


if __name__ == '__main__':
    main()
