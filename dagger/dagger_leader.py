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
    # only one instance of DaggerLeader can be instantiated
    # explicitly use CPU as Tensorflow is buggy when sharing variables on GPU
    device = '/job:ps/task:0/CPU:0'

    max_eps = 5  # max episodes
    train_q_capacity = 2  # max capacity of train_q

    learn_rate = 0.001
    reg_lambda = 1e-4
    checkpoint_eps = 10  # save a checkpoint every 10 episodes

    def __init__(self, cluster, server, worker_tasks):
        self.cluster = cluster
        self.server = server
        self.worker_tasks = worker_tasks

        # original state space and action space
        self.state_dim = Policy.state_dim
        self.action_cnt = Policy.action_cnt
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        self.curr_eps = 0  # current episode, should equal self.eps_cnt

        # create Tensorflow dataflow graph
        self.__create_tf_graph()

# private
    def __create_tf_graph(self):
        # create shared variables on the PS server
        with tf.device(DaggerLeader.device):
            # create a global model on the PS server
            # DaggerLSTM requires a variable scope to collect trainable_vars
            with tf.variable_scope('global'):
                self.global_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                               action_cnt=self.action_cnt)

            # create a shared episode counter used for synchronization
            self.eps_cnt = tf.get_variable(
                'eps_cnt', [], tf.int32,
                initializer=tf.constant_initializer(0))

            # create a shared queue to collect training data from workers
            self.train_q = tf.FIFOQueue(
                capacity=DaggerLeader.train_q_capacity,
                dtypes=[tf.float32, tf.float32],
                shared_name='train_q')  # shared_name is required for sharing

        # op to increment eps_cnt
        self.increment_eps_cnt = self.eps_cnt.assign_add(1)

        # regularization loss
        reg_loss = 0.0
        for var in self.global_model.trainable_vars:
            reg_loss += tf.nn.l2_loss(var)

        # cross entropy loss = rho * soft_CE_loss + (1 - rho) * hard_CE_loss
        self.hard_targets = tf.placeholder(tf.int32, [None, None])
        self.soft_targets = tf.placeholder(tf.float32, [None, None, None])

        hard_ce_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.hard_targets,
                logits=self.global_model.action_scores))

        soft_ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.soft_targets,
                logits=self.global_model.action_scores))

        ce_loss = (1.0 - Config.rho) * hard_ce_loss + Config.rho * soft_ce_loss

        # total loss = cross entropy loss + reg_lambda * regularization loss
        self.total_loss = ce_loss + DaggerLeader.reg_lambda * reg_loss

        # op to train / optimize
        optimizer = tf.train.AdamOptimizer(DaggerLeader.learn_rate)
        self.train_op = optimizer.minimize(self.total_loss)

        # op to summary variables that can be displayed in Tensorboard
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('hard_ce_loss', hard_ce_loss)
        tf.summary.scalar('soft_ce_loss', soft_ce_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()

        # Tensorflow session
        self.sess = tf.Session(
            self.server.target,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def __wait_for_workers(self):
        while True:
            train_q_size = self.sess.run(self.train_q.size())

            if train_q_size == DaggerLeader.train_q_capacity:
                return
            else:
                time.sleep(0.5)

    def __train(self):
        sys.stderr.write('[Leader, Eps {}] train started\n'
                         .format(self.curr_eps))

        while self.sess.run(self.train_q.size()) > 0:
            data = self.sess.run(self.train_q.dequeue())

        sys.stderr.write('[Leader, Eps {}] train ended\n'
                         .format(self.curr_eps))

# public
    def run(self):
        while self.curr_eps < DaggerLeader.max_eps:
            self.curr_eps += 1

            # increment self.eps_cnt to signal workers to start
            self.sess.run(self.increment_eps_cnt)

            # wait until collecting data from all workers
            self.__wait_for_workers()

            # train with collected data
            self.__train()


    def cleanup(self):
        pass
