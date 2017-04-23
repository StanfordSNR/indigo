#!/usr/bin/env python

import os
import sys
import signal
import argparse
from os import path
from subprocess import Popen, call


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

    procs = [None] * len(hostnames)
    for i in xrange(len(hostnames)):
        if i == 0:
            cmd = ['python', args.worker_src,
                   '--ps', args.ps, '--workers', args.workers,
                   '--job-name', 'ps', '--task-index', '0']
        else:
            cmd = ['python', args.worker_src,
                   '--ps', args.ps, '--workers', args.workers,
                   '--job-name', 'worker', '--task-index', str(i - 1)]

        worker_cmd = ['ssh', hostnames[i]] + cmd
        sys.stderr.write('$ %s\n' % ' '.join(worker_cmd))
        procs[i] = Popen(worker_cmd, preexec_fn=os.setsid)

    try:
        procs[0].communicate()
    except:
        pass
    finally:
        sys.stderr.write('\nCleaning up...\n')

        for hostname in hostnames:
            kill_cmd = ['ssh', hostname, 'pkill', '-f', args.worker_src]
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
    parser.add_argument('--worker-src',
                        default='/home/ubuntu/RLCC/a3c/worker.py',
                        help='absolute path to RLCC/a3c/worker.py')
    args = parser.parse_args()

    # run workers
    run(args)


if __name__ == '__main__':
    main()
