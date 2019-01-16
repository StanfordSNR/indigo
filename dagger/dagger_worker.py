#!/usr/bin/env python

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
from collections import Iterable

import context  # noqa # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
from dagger_leader import DaggerLeader
from expert_client import ExpertClient
from helpers.utils import one_hot
from models import DaggerLSTM
from policy import Policy


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env):
        self.cluster = cluster
        self.server = server
        self.task_idx = task_idx
        self.env = env

        # original state space and action space
        self.state_dim = Policy.state_dim
        self.label_dim = Policy.label_dim
        self.action_cnt = Policy.action_cnt
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        # expert policy
        self.expert = ExpertClient()
        env.set_expert_client(self.expert)

        # must call env.set_sample_action() before env.rollout()
        env.set_sample_action(self.sample_action)

        self.curr_eps = 0  # current episode, synchronized with the leader

        # create Tensorflow dataflow graph
        self.__create_tf_graph()

# private
    def __create_tf_graph(self):
        # access shared variables on the PS server
        with tf.device(DaggerLeader.device_sync):
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
                shapes=[[Policy.steps_per_episode, self.aug_state_dim],
                        [Policy.steps_per_episode, self.label_dim]],
                shared_name='train_q')  # shared_name is required for sharing

        # create local variables on this worker
        this_device = '/job:worker/task:{}'.format(self.task_idx)
        with tf.device(this_device):
            # create a local model on this worker
            # DaggerLSTM requires a variable scope to collect trainable_vars
            with tf.variable_scope('local'):
                self.local_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                              action_cnt=self.action_cnt)

        # initial state of LSTM
        self.init_state = self.local_model.zero_init_state(1)

        self.state_data = tf.placeholder(tf.float32, shape=(None, self.aug_state_dim))
        self.action_data = tf.placeholder(tf.float32, shape=(None, self.label_dim))
        # op to enqueue training data
        self.enqueue_train_q = self.train_q.enqueue([self.state_data, self.action_data])

        # op to synchronize the local model with the global model
        local_vars = self.local_model.trainable_vars
        global_vars = self.global_model.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(local_vars, global_vars)])

        # Tensorflow session
        self.sess = tf.Session(
            self.server.target,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def __wait_for_leader(self):
        while True:
            leader_eps_cnt = self.sess.run(self.eps_cnt)
            if leader_eps_cnt == self.curr_eps + 1:
                return
            else:
                time.sleep(0.5)

    def __rollout(self):
        self.state_buf = []
        self.action_buf = []
        self.prev_action = self.action_cnt - 1
        self.lstm_state = self.init_state

        mininet = self.env.env_set[self.env.env_set_idx]
        generator = self.env.tpg_set[self.env.tpg_set_idx_env][self.env.tpg_set_idx_gen]
        generator = generator if type(generator) is str else generator[0]
        sys.stdout.write('[Worker {}, Eps {}]: mininet({:0>2d}/{:0>2d}): {}\n'
                         .format(self.task_idx, self.curr_eps, self.env.env_set_idx+1, len(self.env.env_set), mininet))
        sys.stdout.write('[Worker {}, Eps {}]: traffic({:0>2d}/{:0>2d}): {}\n'
                         .format(self.task_idx, self.curr_eps,
                                 self.__get_flatten_len(self.env.tpg_set[:self.env.tpg_set_idx_env], 2)+self.env.tpg_set_idx_gen+1,
                                 self.__get_flatten_len(self.env.tpg_set, 2), generator))
        sys.stdout.flush()

        if self.env.reset() != -1:
            self.env.rollout()  # will populate self.state_buf and self.action_buf

        self.env.cleanup()

    def __enqueue_data(self):
        # handle the case of no valid data
        if len(self.state_buf) != Policy.steps_per_episode or len(self.action_buf) != Policy.steps_per_episode:
            # feed tensor with fake data
            self.state_buf = [[float('-inf')] * self.aug_state_dim] * Policy.steps_per_episode
            self.action_buf = [[float('-inf')] * self.label_dim] * Policy.steps_per_episode

        # enqueue training data into the training queue
        self.sess.run(self.enqueue_train_q, feed_dict={
            self.state_data: self.state_buf,
            self.action_data: self.action_buf})

        # queue_size = self.sess.run(self.train_q.size())
        # sys.stdout.write(
        #     '[Worker {}, Eps {}]: finished queueing data, '
        #     'current queue size is {}\n'.format
        #     (self.task_idx, self.curr_eps, queue_size))

# public
    def sample_action(self, state):
        # query expert action
        cwnd = state[self.state_dim - 1] * Policy.max_cwnd
        expert_action = self.expert.sample_action(cwnd)
        expert_cwnd = self.expert.best_cwnd

        # construct augmented state
        norm_state = state
        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action

        # fill in state_buf, action_buf
        self.state_buf.append(aug_state)
        self.action_buf.append([cwnd, expert_cwnd, expert_action])

        # always use the expert action on the first episode to get our bearings
        if self.curr_eps == 1:
            self.prev_action = expert_action
            return expert_action

        # feed aug_state into local model and update current LSTM state
        feed_dict = {
            self.local_model.input: [[aug_state]],
            self.local_model.state_in: self.lstm_state,
        }
        ops = [self.local_model.action_probs, self.local_model.state_out]
        action_probs, self.lstm_state = self.sess.run(ops, feed_dict)

        # choose an action to take
        action = np.argmax(action_probs[0][0])
        self.prev_action = action
        return action

    def recursive_flatten(self, iterator):
        ret = []

        for item in iterator:
            if isinstance(item, Iterable):
                ret.extend(self.recursive_flatten(item))
            else:
                ret.append(item)

        return ret

    def __get_flatten_len(self, target, axis):
        flatten_len = 0

        if not isinstance(target, Iterable):
            return 1
        else:
            if axis == 1:
                return len(target)
            else:
                for item in target:
                    flatten_len += self.__get_flatten_len(item, axis-1)

        return flatten_len

    def confirm_sync(self):
        local_vars = self.local_model.trainable_vars
        global_vars = self.global_model.trainable_vars

        check_op = tf.tuple([tf.equal(v1, v2) for v1, v2 in zip(local_vars, global_vars)])

        ret = self.recursive_flatten(self.sess.run(check_op))

        if not reduce(lambda x, y: x and y, ret):
            sys.stdout.write('[Worker {}, Eps {}]: synchronization success\n'
                             .format(self.task_idx, self.curr_eps))
        else:
            sys.stdout.write('[Worker {}, Eps {}]: synchronization failed\n'
                             .format(self.task_idx, self.curr_eps))

    def run(self):
        while self.curr_eps < DaggerLeader.max_eps:
            # start a new episode only if the leader increments eps_cnt
            self.__wait_for_leader()
            self.curr_eps += 1

            if self.curr_eps > 1:
                self.confirm_sync()

            # reset local model to the global model
            self.sess.run(self.sync_op)

            sys.stdout.write('[Worker {}, Eps {}]: rollout started\n'
                             .format(self.task_idx, self.curr_eps))

            while (not self.env.is_all_tasks_done()):
                # populate training data into self.state_buf and self.action_buf
                self.__rollout()
                self.__enqueue_data()

            sys.stdout.write('[Worker {}, Eps {}]: rollout ended\n'
                             .format(self.task_idx, self.curr_eps))
            sys.stdout.flush()

    def cleanup(self):
        self.env.cleanup()
