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

import datetime
import math
import random
import sys
import time
from os import path

import context
import numpy as np
import tensorflow as tf
from helpers.utils import Config, make_sure_path_exists, min_x_max, softmax, timestamp_ms
from models import DaggerLSTM
from policy import Policy


class DaggerLeader(object):
    # only one instance of DaggerLeader can be instantiated
    # explicitly use CPU as Tensorflow is buggy when sharing variables on GPU
    device_sync = '/job:ps/task:0/CPU:0'
    device_train = '/job:ps/task:0/{}:0'.format(Config.device)

    max_eps = 5000  # max episodes
    train_q_capacity = Config.total_tpg_num_train  # max capacity of train_q
    default_batch_size = Config.batch_size

    learn_rate = 0.001
    reg_lambda = 1e-4
    checkpoint_eps = 10  # save a checkpoint every 10 episodes

    def __init__(self, cluster, server, worker_tasks):
        self.cluster = cluster  # unused
        self.server = server
        self.worker_tasks = worker_tasks  # unused

        # original state space and action space
        self.state_dim = Policy.state_dim
        self.label_dim = Policy.label_dim
        self.action_cnt = Policy.action_cnt
        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        # batched data buffers
        self.aggregated_states = []
        self.aggregated_actions = []
        self.aggregated_soft_targets = []

        self.curr_eps = 0  # current episode, should equal self.eps_cnt
        self.summary_step = 0

        self.shuffle_cwnd = Config.total_tpg_num_train * Config.shuffle_window

        # logger
        self.logdir = path.join(context.base_dir, 'dagger', 'logs',
                                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        make_sure_path_exists(self.logdir)

        # create Tensorflow dataflow graph
        self.__create_tf_graph()

# private
    def __create_tf_graph(self):
        # create trainable variables on the PS server
        with tf.device(DaggerLeader.device_train):
            with tf.variable_scope('local'):
                self.local_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                              action_cnt=self.action_cnt)
        # create shared variables on the PS server
        with tf.device(DaggerLeader.device_sync):
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
                shapes=[[Policy.steps_per_episode, self.aug_state_dim],
                        [Policy.steps_per_episode, self.label_dim]],
                shared_name='train_q')  # shared_name is required for sharing

        # default initial state of LSTM
        self.default_init_state = self.local_model.zero_init_state(
            DaggerLeader.default_batch_size)

        # op to increment eps_cnt
        self.increment_eps_cnt = self.eps_cnt.assign_add(1)

        # regularization loss
        reg_loss = 0.0
        for var in self.local_model.trainable_vars:
            reg_loss += tf.nn.l2_loss(var)
        reg_loss *= DaggerLeader.reg_lambda

        # cross entropy loss = rho * soft_CE_loss + (1 - rho) * hard_CE_loss
        self.hard_targets = tf.placeholder(tf.int32, [None, None])
        self.soft_targets = tf.placeholder(tf.float32, [None, None, None])

        hard_ce_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.hard_targets,
                logits=self.local_model.action_scores))

        soft_ce_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.soft_targets,
                logits=self.local_model.action_scores))

        ce_loss = (1.0 - Config.rho) * hard_ce_loss + Config.rho * soft_ce_loss

        # total loss = cross entropy loss + reg_lambda * regularization loss
        self.total_loss = ce_loss + reg_loss

        # op to train / optimize
        optimizer = tf.train.AdamOptimizer(DaggerLeader.learn_rate)
        self.train_op = optimizer.minimize(self.total_loss)

        # op to synchronize the global model with the local model
        local_vars = self.local_model.trainable_vars
        global_vars = self.global_model.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(global_vars, local_vars)])

        # op to summary variables that can be displayed in Tensorboard
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('hard_ce_loss', hard_ce_loss)
        tf.summary.scalar('soft_ce_loss', soft_ce_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logdir)

        # Tensorflow session
        self.sess = tf.Session(
            self.server.target,
            config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def __save_model(self, checkpoint=None):
        if checkpoint is None:
            model_path = path.join(self.logdir, 'model')
        else:
            model_path = path.join(
                self.logdir, 'checkpoint-{}'.format(checkpoint))

        # save parameters to parameter server
        saver = tf.train.Saver(self.local_model.trainable_vars)
        saver.save(self.sess, model_path)
        sys.stdout.write(
            'model saved to param server at {}\n'.format(model_path))

    def __wait_for_workers(self):
        estimated_time = (DaggerLeader.train_q_capacity * Policy.steps_per_episode
                          * Policy.min_step_len / 1000.0 / len(self.worker_tasks))
        time.sleep(estimated_time)

        while True:
            train_q_size = self.sess.run(self.train_q.size())

            if train_q_size == DaggerLeader.train_q_capacity:
                return
            else:
                time.sleep(10)

    def __get_soft_target(self, curr_cwnds, best_cwnds):
        soft_targets = np.zeros([len(curr_cwnds), self.action_cnt], np.float32)
        for i in range(len(curr_cwnds)):
            curr_cwnd = curr_cwnds[i]
            best_cwnd = best_cwnds[i]

            # calculate the distance between
            # best_cwnd and (current_cwnd op action)
            soft_cwnds = []
            for op, val in Policy.action_mapping:
                soft_cwnds.append(min_x_max(Policy.min_cwnd, op(curr_cwnd, val), Policy.max_cwnd))
            action_distance = -np.abs(np.array(soft_cwnds) - best_cwnd)

            # transform to Gaussian distribution
            if np.std(action_distance) == 0.0:
                normalized_distance = action_distance
            else:
                normalized_distance = (
                    (action_distance - np.mean(action_distance)) /
                    np.std(action_distance))
            soft_targets[i] = softmax(normalized_distance)
        return soft_targets

    def __dequeue_train_q(self):
        aug_states, aug_labels = self.sess.run(self.train_q.dequeue_many(DaggerLeader.train_q_capacity))
        for i in xrange(DaggerLeader.train_q_capacity):
            states = aug_states[i]
            labels = aug_labels[i]

            if math.isinf(states[0][0]):
                continue

            curr_cwnd, best_cwnd, best_action = np.array(labels).T

            self.aggregated_states.append(states)
            self.aggregated_actions.append(best_action)
            self.aggregated_soft_targets.append(
                self.__get_soft_target(curr_cwnd, best_cwnd))

    def __train(self):
        # data shuffle
        aggregated_data = zip(self.aggregated_states[-1*self.shuffle_cwnd:],
                              self.aggregated_actions[-1*self.shuffle_cwnd:],
                              self.aggregated_soft_targets[-1*self.shuffle_cwnd:])
        random.shuffle(aggregated_data)
        shuffled_states, shuffled_actions, shuffled_soft_targets = zip(*aggregated_data)

        batch_size = min(len(shuffled_states), self.default_batch_size)
        num_batches = len(shuffled_states) / batch_size

        # set len of init state equal to batch_size
        if batch_size != self.default_batch_size:
            self.init_state = self.local_model.zero_init_state(batch_size)
        else:
            self.init_state = self.default_init_state

        for curr_epoch in xrange(50):
            mean_loss = 0.0
            max_loss = 0.0
            start_ts = timestamp_ms()
            for batch_num in xrange(num_batches):
                start = batch_num * batch_size
                end = start + batch_size

                batch_states = shuffled_states[start:end]
                batch_soft_targets = shuffled_soft_targets[start:end]
                batch_actions = shuffled_actions[start:end]

                ops_to_run = [self.train_op, self.total_loss, self.summary_op]
                train_ret, loss, summary = self.sess.run(ops_to_run, feed_dict={
                    self.local_model.input: batch_states,
                    self.soft_targets: batch_soft_targets,
                    self.hard_targets: batch_actions,
                    self.local_model.state_in: self.init_state})

                mean_loss += loss
                max_loss = max(loss, max_loss)

                self.summary_writer.add_summary(summary, self.summary_step)
                self.summary_step += 1

            elapsed = (timestamp_ms() - start_ts) / 1000.0
            mean_loss /= num_batches
            sys.stdout.write('--- epoch %d: max loss %.4f, mean loss %.4f, time cost %.4f\n' %
                             (curr_epoch, max_loss, mean_loss, elapsed))
            sys.stdout.flush()

# public
    def run(self):
        while self.curr_eps < DaggerLeader.max_eps:
            self.curr_eps += 1

            # wait for gRPC established and sync
            while len(self.sess.run(tf.report_uninitialized_variables())) > 0:
                time.sleep(1.0)

            # increment self.eps_cnt to signal workers to start
            self.sess.run(self.increment_eps_cnt)

            sys.stdout.write('[Leader {}, Eps {}]: start waiting for worker\n'
                             .format(self.server.server_def.task_index, self.curr_eps))
            sys.stdout.flush()
            # wait until having collected data from all workers
            self.__wait_for_workers()

            # dequeue collected data from train_q
            self.__dequeue_train_q()

            sys.stdout.write('[Leader {}, Eps {}]: start to train\n'
                             .format(self.server.server_def.task_index, self.curr_eps))
            sys.stdout.flush()
            # train
            self.__train()

            # copy model para from local model to global model
            self.sess.run(self.sync_op)

            sys.stdout.write('[Leader {}, Eps {}]: finished training\n\n'
                             .format(self.server.server_def.task_index, self.curr_eps))
            sys.stdout.flush()

            # save model every 'checkpoint_eps' episodes
            if self.curr_eps % self.checkpoint_eps == 0:
                self.__save_model(self.curr_eps)

    def cleanup(self):
        self.__save_model()
