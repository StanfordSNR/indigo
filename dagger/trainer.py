#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Wei Wang, Yiyang Shao (Huawei Technologies)
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


import argparse
import sys

import context
import tensorflow as tf
from dagger_leader import DaggerLeader
from dagger_worker import DaggerWorker
from env.mininet_env import MininetEnv
from helpers.utils import Config


def create_mininet_env(worker_cnt, worker_index):
    # settings of emulated networks (env) and background traffic patterns (tp)
    total_env_set = Config.total_env_set_train
    total_tpg_set = Config.total_tpg_set_train
    total_env_len = len(total_env_set)
    tasks_per_worker = total_env_len / worker_cnt

    # allocate env and tp to this worker
    this_env_set = []
    this_tpg_set = []

    if worker_index < worker_cnt - 1:
        for i in xrange(tasks_per_worker * worker_index,
                        tasks_per_worker * (worker_index + 1)):
            this_env_set.append(total_env_set[i])
            this_tpg_set.append(total_tpg_set[i])

            sys.stderr.write('worker {} is allocated with env {} and tpg {}\n'
                             .format(worker_index, total_env_set[i], total_tpg_set[i]))
    else:  # last worker
        for i in xrange(tasks_per_worker * worker_index, total_env_len):
            this_env_set.append(total_env_set[i])
            this_tpg_set.append(total_tpg_set[i])

            sys.stderr.write('worker {} is allocated with env {} and tpg {}\n'
                             .format(worker_index, total_env_set[i], total_tpg_set[i]))

    env = MininetEnv(this_env_set, this_tpg_set, True)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ps-hosts', metavar='[HOSTNAME:PORT, ...]', required=True,
                        help='comma-separated list of hostname:port of parameter servers')
    parser.add_argument('--worker-hosts', metavar='[HOSTNAME:PORT, ...]', required=True,
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
        env = create_mininet_env(num_workers, task_index)
        worker = DaggerWorker(cluster, server, task_index, env)

        try:
            worker.run()
        except KeyboardInterrupt:
            pass
        finally:
            worker.cleanup()


if __name__ == '__main__':
    main()
