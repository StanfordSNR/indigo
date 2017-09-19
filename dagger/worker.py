#!/usr/bin/env python

import sys
import yaml
import argparse
import project_root
import numpy as np
import tensorflow as tf
import logging
from subprocess import check_call
import multiprocessing
from os import path
from dagger import DaggerLeader, DaggerWorker
from env.environment import Environment
from env.sender import Sender


def prepare_traces(bandwidth):
    trace_dir = path.join(project_root.DIR, 'env')

    if type(bandwidth) == int or type(bandwidth) == float:
        trace_path = path.join(trace_dir, '%smbps.trace' % str(bandwidth))

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


def create_env(host_index, num_flows, start_worker, end_worker):
    """ Creates and returns an Environment which contains a single
    sender-receiver connection. The environment is run inside mahimahi
    shells. The environment knows the best cwnd to pass to the expert policy.
    """

    if host_index <= 2:
        trace_path = path.join(project_root.DIR, 'env', '100.42mbps.trace')
        mm_cmd = ('mm-delay 27 mm-link %s %s --uplink-queue=droptail '
                  '--uplink-queue-args=packets=173') % (trace_path, trace_path)
        best_cwnd = 500
    elif host_index <= 5:
        trace_path = path.join(project_root.DIR, 'env', '77.72mbps.trace')
        mm_cmd = ('mm-delay 51 mm-loss uplink 0.0006 mm-link %s %s '
                  '--uplink-queue=droptail '
                  '--uplink-queue-args=packets=94') % (trace_path, trace_path)
        best_cwnd = 690
    elif host_index <= 8:
        trace_path = path.join(project_root.DIR, 'env', '114.68mbps.trace')
        mm_cmd = ('mm-delay 45 mm-link %s %s --uplink-queue=droptail '
                  '--uplink-queue-args=packets=450') % (trace_path, trace_path)
        best_cwnd = 870

    env = Environment(mm_cmd, num_flows, start_worker, end_worker, best_cwnd)
    return env


def run_worker(ps, workers, job, num_hosts, num_flows,
               worker_idx, env, in_charge, ports_q=None, flow_info_q=None):
    try:
        cluster = tf.train.ClusterSpec({'ps': ps, 'worker': workers})
        server = tf.train.Server(cluster, job_name=job, task_index=worker_idx)

        worker = DaggerWorker(cluster, server, worker_idx,
                              num_hosts, num_flows, env,
                              in_charge, ports, flow_info_q)
        worker.run(debug=True)
        worker.cleanup()
    except KeyboardInterrupt:
        pass


def make_multiple_hosts(hosts, flows):
    """ Given host names in the form of IP:port and list of # flows,
    return a list of length [sum of flows] of hosts
    """
    new_hosts = []
    next_port = int(hosts[0].split(':')[1])

    for i in xrange(len(hosts)):
        ip = hosts[i].split(':')[0]

        for i in xrange(flows[i]):
            new_hosts.append(ip + ':' + str(next_port))
            next_port += 1

    return new_hosts


def run(args):
    """ For each worker/parameter server, starts the appropriate job
    associated with the cluster and server.
    """

    flows = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    num_hosts = len(flows)

    job_name = args.job_name
    host_index = args.host_index
    sys.stderr.write('Starting job %s host %d\n' % (job_name, host_index))

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    assert num_hosts <= len(worker_hosts), 'Not enough hosts to run!'
    workers = make_multiple_hosts(worker_hosts, flows)

    if job_name == 'ps':
        # Sets up the queue, shared variables, and global classifier.
        cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': workers})
        server = tf.train.Server(cluster, job_name=job_name,
                                 task_index=host_index)
        worker_tasks = set([idx for idx in xrange(len(workers))])
        leader = DaggerLeader(cluster, server, num_hosts, worker_tasks)

        try:
            leader.run(debug=True)
        except KeyboardInterrupt:
            pass
        finally:
            leader.cleanup()

    elif job_name == 'worker':
        # Sets up the env, shared variables (sync, classifier, queue, etc)
        num_flows = flows[host_index]
        end_worker = sum(flows[:host_index+1]) - 1
        start_worker = end_worker - num_flows + 1
        env = create_env(host_index, num_flows, start_worker, end_worker)

        if num_flows == 1:      # Don't need multiprocessing for single flow
            run_worker(ps_hosts, workers, job_name, num_hosts, num_flows,
                       start_worker, env, True)
        else:

            # TODO extra: in-charge flow is the only one that can gather
            # data. Differentiate between in-charge and able to gather data.

            pool = multiprocessing.Pool(num_flows)
            results = []
            manager = multiprocessing.Manager()

            # In-charge flow gives start time and # steps to other flows
            flow_info_q = manager.Queue(num_flows-1)
            # Other flows share ports and flow info with the one in charge
            ports_q = manager.Queue(num_flows-1)

            for worker_idx in xrange(start_worker, end_worker+1):
                in_charge = worker_idx == start_worker
                result = pool.apply_async(
                        run_worker,
                        args=(ps_hosts, workers, job_name, num_hosts,
                              num_flows, worker_idx, env,
                              in_charge, ports_q, flow_info_q))

                results.append(result)

            pool.close()
            for res in results:     # Re-raise any exceptions
                res.get()
            pool.join()


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
    parser.add_argument('--host-index', metavar='N', type=int, required=True,
                        help='index of host')
    args = parser.parse_args()

    # run parameter servers and workers
    run(args)


if __name__ == '__main__':
    main()
