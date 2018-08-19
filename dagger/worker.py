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


import sys
import yaml
import argparse
import project_root
import numpy as np
import tensorflow as tf
from subprocess import check_call
from os import path
from dagger import DaggerLeader, DaggerWorker
from env.environment import Environment
from env.sender import Sender


def prepare_traces(bandwidth):
    trace_dir = path.join(project_root.DIR, 'env')

    if type(bandwidth) == int:
        trace_path = path.join(trace_dir, '%dmbps.trace' % bandwidth)

        if not path.exists(trace_path):
            gen_trace = path.join(project_root.DIR, 'helpers',
                                  'generate_trace.py')
            cmd = ['python', gen_trace, '--output-dir', trace_dir,
                   '--bandwidth', str(bandwidth)]
            sys.stderr.write('$ %s\n' % ' '.join(cmd))
            check_call(cmd)

        uplink_trace = trace_path
        downlink_trace = uplink_trace
    else:
        trace_path = path.join(trace_dir, bandwidth)
        # intentionally switch uplink and downlink traces due to sender first
        uplink_trace = trace_path + '.down'
        downlink_trace = trace_path + '.up'

    return uplink_trace, downlink_trace


def create_env(task_index):
    """ Creates and returns an Environment which contains a single
    sender-receiver connection. The environment is run inside mahimahi
    shells. The environment knows the best cwnd to pass to the expert policy.
    """

    best_cwnds_file = path.join(project_root.DIR, 'dagger', 'best_cwnds.yml')
    best_cwnd_map = yaml.load(open(best_cwnds_file))

    if task_index == 0:
        trace_path = path.join(project_root.DIR, 'env', '0.57mbps-poisson.trace')
        mm_cmd = 'mm-delay 28 mm-loss uplink 0.0477 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=14' % (trace_path, trace_path)
        best_cwnd = 5
    elif task_index == 1:
        trace_path = path.join(project_root.DIR, 'env', '2.64mbps-poisson.trace')
        mm_cmd = 'mm-delay 88 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=130' % (trace_path, trace_path)
        best_cwnd = 40
    elif task_index == 2:
        trace_path = path.join(project_root.DIR, 'env', '3.04mbps-poisson.trace')
        mm_cmd = 'mm-delay 130 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=426' % (trace_path, trace_path)
        best_cwnd = 70
    elif task_index <= 18:
        bandwidth = [5, 10, 20, 50]
        delay = [10, 20, 40, 80]

        cartesian = [(b, d) for b in bandwidth for d in delay]
        bandwidth, delay = cartesian[task_index - 3]

        uplink_trace, downlink_trace = prepare_traces(bandwidth)
        mm_cmd = 'mm-delay %d mm-link %s %s' % (delay, uplink_trace, downlink_trace)
        best_cwnd = best_cwnd_map[bandwidth][delay]
    elif task_index == 19:
        trace_path = path.join(project_root.DIR, 'env', '100.42mbps.trace')
        mm_cmd = 'mm-delay 27 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=173' % (trace_path, trace_path)
        best_cwnd = 500
    elif task_index == 20:
        trace_path = path.join(project_root.DIR, 'env', '77.72mbps.trace')
        mm_cmd = 'mm-delay 51 mm-loss uplink 0.0006 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=94' % (trace_path, trace_path)
        best_cwnd = 690
    elif task_index == 21:
        trace_path = path.join(project_root.DIR, 'env', '114.68mbps.trace')
        mm_cmd = 'mm-delay 45 mm-link %s %s --uplink-queue=droptail --uplink-queue-args=packets=450' % (trace_path, trace_path)
        best_cwnd = 870
    elif task_index <= 29:
        bandwidth = [100, 200]
        delay = [10, 20, 40, 80]

        cartesian = [(b, d) for b in bandwidth for d in delay]
        bandwidth, delay = cartesian[task_index - 26]

        uplink_trace, downlink_trace = prepare_traces(bandwidth)
        mm_cmd = 'mm-delay %d mm-link %s %s' % (delay, uplink_trace, downlink_trace)
        best_cwnd = best_cwnd_map[bandwidth][delay]

    env = Environment(mm_cmd)
    env.best_cwnd = best_cwnd

    return env


def run(args):
    """ For each worker/parameter server, starts the appropriate job
    associated with the cluster and server.
    """

    job_name = args.job_name
    task_index = args.task_index
    sys.stderr.write('Starting job %s task %d\n' % (job_name, task_index))

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    num_workers = len(worker_hosts)

    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    if job_name == 'ps':
        # Sets up the queue, shared variables, and global classifier.
        worker_tasks = set([idx for idx in xrange(num_workers)])
        leader = DaggerLeader(cluster, server, worker_tasks)
        try:
            leader.run(debug=True)
        except KeyboardInterrupt:
            pass
        finally:
            leader.cleanup()

    elif job_name == 'worker':
        # Sets up the env, shared variables (sync, classifier, queue, etc)
        env = create_env(task_index)
        learner = DaggerWorker(cluster, server, task_index, env)
        try:
            learner.run(debug=True)
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
