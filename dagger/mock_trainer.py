#!/usr/bin/env python

import sys
import argparse
import tensorflow as tf

import context
from dagger_leader import DaggerLeader
from dagger_worker import DaggerWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ps-hosts', metavar='[HOSTNAME:PORT, ...]', required=True,
        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument(
        '--worker-hosts', metavar='[HOSTNAME:PORT, ...]', required=True,
        help='comma-separated list of hostname:port of workers')
    parser.add_argument('--job-name', choices=['ps', 'worker'],
                        required=True, help='ps or worker')
    parser.add_argument('--task-index', metavar='N', type=int, required=True,
                        help='index of task')
    args = parser.parse_args()

    job_name = args.job_name
    task_index = args.task_index
    sys.stderr.write('Starting job {} task {}\n'.format(job_name, task_index))

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    num_workers = len(worker_hosts)

    # start the appropriate job associated with tensorflow cluster and server
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        worker_tasks = set([idx for idx in xrange(num_workers)])
        leader = DaggerLeader(cluster, server, worker_tasks)

        try:
            leader.run()
        except KeyboardInterrupt:
            pass
        finally:
            leader.cleanup()

    elif job_name == 'worker':
        env = None
        worker = DaggerWorker(cluster, server, task_index, env)

        try:
            worker.run()
        except KeyboardInterrupt:
            pass
        finally:
            worker.cleanup()


if __name__ == '__main__':
    main()
