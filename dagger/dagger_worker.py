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
import random
import numpy as np
import tensorflow as tf

import context
from policy import Policy
from models import DaggerLSTM
from experts import ExpertClient
from dagger_leader import DaggerLeader
from helpers.utils import make_sure_path_exists, one_hot, Config


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env):
        self.cluster = cluster
        self.server = server
        self.task_idx = task_idx
        self.env = env

        # original state space and action space
        self.state_dim = Policy.state_dim
        self.action_cnt = Policy.action_cnt
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        self.curr_eps = 0  # current episode, synchronized with the leader

        # create Tensorflow dataflow graph
        self.__create_tf_graph()

# private
    def __create_tf_graph(self):
        # access shared variables on the PS server
        with tf.device(DaggerLeader.device):
            # access the global model on the PS server
            with tf.variable_scope('global'):
                self.global_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                               action_cnt=self.action_cnt)

            # access the shared episode counter used for synchronization
            self.eps_cnt = tf.get_variable(
                'eps_cnt', [], tf.int32,
                initializer=tf.constant_initializer(0))

            # access the shared queue to store training data
            self.train_q = tf.FIFOQueue(
                capacity=DaggerLeader.train_q_capacity,
                dtypes=[tf.float32, tf.float32],
                shared_name='train_q')  # shared_name is required for sharing

        # create local variables on this worker
        this_device = '/job:worker/task:{}'.format(self.task_idx)
        with tf.device(this_device):
            # create a local model on this worker
            # DaggerLSTM requires a variable scope to collect trainable_vars
            with tf.variable_scope('local'):
                self.local_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                              action_cnt=self.action_cnt)

        # op to enqueue training data to train_q
        self.enqueue_train_q = self.train_q.enqueue([1.0, 2.0])

        # op to synchronize the local model with the global model
        local_vars = self.local_model.trainable_vars
        global_vars = self.global_model.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2)
                                  for v1, v2 in zip(local_vars, global_vars)])

        # Tensorflow session
        self.sess = tf.Session(
            self.server.target,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def __wait_for_leader(self):
        while True:
            leader_eps_cnt = self.sess.run(self.eps_cnt)

            if leader_eps_cnt == self.curr_eps + 1:
                self.curr_eps = leader_eps_cnt
                return
            else:
                time.sleep(0.5)

    def __work(self):
        sys.stderr.write('[Worker {}, Eps {}] work started\n'
                         .format(self.task_idx, self.curr_eps))
        time.sleep(random.randint(1, 5))
        self.sess.run(self.enqueue_train_q)
        sys.stderr.write('[Worker {}, Eps {}] work ended\n'
                         .format(self.task_idx, self.curr_eps))

# public
    def run(self):
        while self.curr_eps < DaggerLeader.max_eps:
            # increment curr_eps only if the leader has incremented it
            self.__wait_for_leader()

            # collect training data
            self.__work()


    def cleanup(self):
        pass
