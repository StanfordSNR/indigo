#!/usr/bin/env python

import sys
import argparse
import project_root
import tensorflow as tf
from a3c import A3C
from env.sender import Sender


class Worker(object):
    def __init__(self, args):
        self.ps_hosts = args.ps_hosts
        self.worker_hosts = args.worker_hosts
        self.job_name = args.job_name
        self.task_index = args.task_index
        self.port = args.port

        self.max_episodes = 2000

    def work(self):
        self.sender = Sender(self.port, training=True)

        self.learner = A3C(
            cluster=self.cluster,
            server=self.server,
            worker_device='/job:worker/task:%d' % self.task_index,
            state_dim=self.sender.state_dim,
            action_cnt=self.sender.action_cnt,
            debug=True)

        self.sender.set_sample_action(self.learner.sample_action)

        for i in xrange(1, self.max_episodes):
            self.sender.run()

            state_buf, action_buf, reward = self.sender.get_experience()
            self.learner.update_model(state_buf, action_buf, reward)

            self.sender.reset()

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
    parser.add_argument('--port', type=int, help='port of sender to train')
    args = parser.parse_args()

    sys.stderr.write('Starting job %s task %d\n' %
                     (args.job_name, args.task_index))

    worker = Worker(args)
    worker.run()


if __name__ == '__main__':
    main()
