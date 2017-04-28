#!/usr/bin/env python

import sys
import argparse
import project_root
import tensorflow as tf
from os import path
from a3c import A3C
from env.environment import Environment


class Worker(object):
    def __init__(self, args):
        self.ps_hosts = args.ps_hosts
        self.worker_hosts = args.worker_hosts
        self.job_name = args.job_name
        self.task_index = args.task_index

    def create_env(self):
        uplink_trace = path.join(project_root.DIR, 'env', '12mbps.trace')
        downlink_trace = uplink_trace
        mm_cmd = (
            'mm-delay 20 mm-link %s %s '
            '--uplink-queue=droptail --uplink-queue-args=packets=200' %
            (uplink_trace, downlink_trace))

        env = Environment(mm_cmd)
        env.setup()
        return env

    def work(self):
        env = self.create_env()

        learner = A3C(
            cluster=self.cluster,
            server=self.server,
            worker_device='/job:worker/task:%d' % self.task_index,
            env=env)

        try:
            learner.run()
        except KeyboardInterrupt:
            pass
        finally:
            learner.cleanup()

    def run(self):
        ps_hosts = self.ps_hosts.split(',')
        worker_hosts = self.worker_hosts.split(',')

        self.cluster = tf.train.ClusterSpec(
            {'ps': ps_hosts, 'worker': worker_hosts})
        self.server = tf.train.Server(
            self.cluster, job_name=self.job_name, task_index=self.task_index)

        if self.job_name == 'ps':
            self.server.join()
        elif self.job_name == 'worker':
            self.work()


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

    sys.stderr.write('Starting job %s task %d\n' %
                     (args.job_name, args.task_index))

    worker = Worker(args)
    worker.run()


if __name__ == '__main__':
    main()
