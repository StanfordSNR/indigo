#!/usr/bin/env python

import sys
import argparse
import project_root
import tensorflow as tf
from os import path
from a3c import A3C
from env.environment import Environment


def create_env(task_index):
    uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
    downlink_trace = uplink_trace
    mm_cmd = (
        'mm-delay 20 mm-link %s %s '
        '--uplink-queue=droptail --uplink-queue-args=packets=200' %
        (uplink_trace, downlink_trace))

    env = Environment(mm_cmd)
    env.setup()
    return env


def run(args):
    job_name = args.job_name
    task_index = args.task_index
    sys.stderr.write('Starting job %s task %d\n' % (job_name, task_index))

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        server.join()
    elif job_name == 'worker':
        env = create_env(task_index)
        learner = A3C(
            cluster=cluster,
            server=server,
            task_index=task_index,
            env=env)

        try:
            learner.run()
        except KeyboardInterrupt:
            pass
        finally:
            learner.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ps-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument(
        '--worker-hosts', required=True, metavar='[HOSTNAME:PORT, ...]',
        help='comma-separated list of hostname:port of workers')
    parser.add_argument('--job-name', choices=['ps', 'worker'],
                        required=True, help='ps or worker')
    parser.add_argument('--task-index', metavar='N', type=int, required=True,
                        help='index of task')
    args = parser.parse_args()

    # run parameter servers and workers
    run(args)


if __name__ == '__main__':
    main()
