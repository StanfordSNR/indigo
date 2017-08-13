#!/usr/bin/env python

import yaml
import sys
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

    bandwidth = [10, 40, 70, 100]
    delay = [10, 40, 70, 100]
    queue = None

    cartesian = [(b,d) for b in bandwidth for d in delay]
    bandwidth, delay = cartesian[task_index]
    uplink_trace, downlink_trace = prepare_traces(bandwidth)

    mm_cmd = ('mm-delay %d mm-link %s %s' %
              (delay, uplink_trace, downlink_trace))
    if queue is not None:
        mm_cmd += (' --downlink-queue=droptail '
                   '--downlink-queue-args=packets=%d' % queue)

    env = Environment(mm_cmd)
    env.setup()
    
    if delay == 25:
        env.best_cwnd = bandwidth * 5
    else:
        cwnds_file = path.join(project_root.DIR, 'dagger', 'best_cwnds.yml')
        env.best_cwnd = yaml.load(open(cwnds_file))[bandwidth][delay]

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
