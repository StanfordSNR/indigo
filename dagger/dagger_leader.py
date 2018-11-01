# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Yiyang Shao, Wei Wang (Huawei Technologies)
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
import time
import datetime
import socket
import numpy as np
import tensorflow as tf
from subprocess import check_output
from os import path

import context
from policy import Policy
from models import DaggerLSTM
from helpers.utils import make_sure_path_exists, timestamp_ms, softmax, Config


class DaggerLeader(object):
    learn_rate = 0.001
    reg_lambda = 1e-4
    checkpoint_eps = 10  # save a checkpoint every 10 episodes

    max_eps = 5  # max episodes
    train_q_capacity = 2

    def __init__(self, cluster, server, worker_tasks):
        self.cluster = cluster
        self.server = server
        self.worker_tasks = worker_tasks

        # original state space and action space
        self.state_dim = Policy.state_dim
        self.action_cnt = Policy.action_cnt
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        self.curr_eps = 0  # current episode, equals self.eps_cnt

        # create Tensorflow dataflow graph
        self.__create_tf_graph()

# private
    def __create_tf_graph(self):
        # create shared variables on the PS server
        # GPU is prioritized if available
        #with tf.device(tf.train.replica_device_setter(cluster=self.cluster)):
        with tf.device('/job:ps/task:0/CPU:0'):
            # create a shared episode counter used for synchronization
            self.eps_cnt = tf.get_variable(
                'eps_cnt', [], tf.float32,
                initializer=tf.constant_initializer(0))
            self.increment_eps_cnt = self.eps_cnt.assign_add(1)

            # create a shared queue to collect training data from workers
            self.train_q = tf.FIFOQueue(
                capacity=DaggerLeader.train_q_capacity,  # TODO: test only
                dtypes=[tf.float32, tf.float32],
                shared_name='train_q')  # unique shared_name is required
            self.enqueue_train_q = self.train_q.enqueue([1.0, 2.0])  # TODO: dummy data

        self.sess = tf.Session(
            self.server.target,
            config=tf.ConfigProto(log_device_placement=True,
                                  allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def __wait_for_workers(self):
        while True:
            train_q_size = self.sess.run(self.train_q.size())
            print 'train_q size', train_q_size

            if train_q_size == DaggerLeader.train_q_capacity:
                return
            else:
                time.sleep(0.5)

    def __train(self):
        sys.stderr.write('[Leader] Episode {}: train started\n'
                         .format(self.curr_eps))
        time.sleep(3)
        sys.stderr.write('[Leader] Episode {}: train ended\n'
                         .format(self.curr_eps))

# public
    def run(self):
        while self.curr_eps < DaggerLeader.max_eps:
            self.curr_eps += 1
            sys.stderr.write('[Leader] Episode {} starts\n'
                             .format(self.curr_eps))

            # increment self.eps_cnt to signal workers to start
            self.sess.run(self.increment_eps_cnt)
            print 'leader eps', self.sess.run(self.eps_cnt)

            # wait until collecting data from all workers
            self.__wait_for_workers()

            while self.sess.run(self.train_q.size()) > 0:
                data = self.sess.run(self.train_q.dequeue())
                print data

            # train with collected data
            self.__train()


    def cleanup(self):
        pass
