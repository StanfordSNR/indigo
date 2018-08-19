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
import time
import project_root
import numpy as np
import tensorflow as tf
import datetime
from tensorflow import contrib
from os import path
from models import DaggerLSTM
from experts import TrueDaggerExpert
from env.sender import Sender
from helpers.helpers import (
    make_sure_path_exists, normalize, one_hot, curr_ts_ms)
from subprocess import check_output


class Status:
    EP_DONE = 0
    WORKER_DONE = 1
    WORKER_START = 2
    PS_DONE = 3


class DaggerLeader(object):
    def __init__(self, cluster, server, worker_tasks):
        self.cluster = cluster
        self.server = server
        self.worker_tasks = worker_tasks
        self.num_workers = len(worker_tasks)
        self.aggregated_states = []
        self.aggregated_actions = []
        self.max_eps = 1000
        self.checkpoint_delta = 10
        self.checkpoint = self.checkpoint_delta
        self.learn_rate = 0.01
        self.regularization_lambda = 1e-4
        self.train_step = 0

        self.state_dim = Sender.state_dim
        self.action_cnt = Sender.action_cnt
        self.aug_state_dim = self.state_dim + self.action_cnt

        # Create the master network and training/sync queues
        with tf.variable_scope('global'):
            self.global_network = DaggerLSTM(
                state_dim=self.aug_state_dim, action_cnt=self.action_cnt)

        self.leader_device_cpu = '/job:ps/task:0/cpu:0'
        with tf.device(self.leader_device_cpu):
            with tf.variable_scope('global_cpu'):
                self.global_network_cpu = DaggerLSTM(
                    state_dim=self.aug_state_dim, action_cnt=self.action_cnt)

        cpu_vars = self.global_network_cpu.trainable_vars
        gpu_vars = self.global_network.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            cpu_vars, gpu_vars)])

        self.default_batch_size = 300
        self.default_init_state = self.global_network.zero_init_state(
                self.default_batch_size)

        # Each element is [[aug_state]], [action]
        self.train_q = tf.FIFOQueue(
                self.num_workers, [tf.float32, tf.int32],
                shared_name='training_feed')

        # Keys: worker indices, values: Tensorflow messaging queues
        # Queue Elements: Status message
        self.sync_queues = {}
        for idx in worker_tasks:
            queue_name = 'sync_q_%d' % idx
            self.sync_queues[idx] = tf.FIFOQueue(3, [tf.int16],
                                                 shared_name=queue_name)

        self.setup_tf_ops(server)

        self.sess = tf.Session(
            server.target, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        """ Sends messages to workers to stop and saves the model. """
        for idx in self.worker_tasks:
            self.sess.run(self.sync_queues[idx].enqueue(Status.PS_DONE))
        self.save_model()

    def save_model(self, checkpoint=None):
        """ Takes care of saving/checkpointing the model. """
        if checkpoint is None:
            model_path = path.join(self.logdir, 'model')
        else:
            model_path = path.join(self.logdir, 'checkpoint-%d' % checkpoint)

        # save parameters to parameter server
        saver = tf.train.Saver(self.global_network.trainable_vars)
        saver.save(self.sess, model_path)
        sys.stderr.write('\nModel saved to param. server at %s\n' % model_path)

    def setup_tf_ops(self, server):
        """ Sets up Tensorboard operators and tools, such as the optimizer,
        summary values, Tensorboard, and Session.
        """

        self.actions = tf.placeholder(tf.int32, [None, None])

        reg_loss = 0.0
        for x in self.global_network.trainable_vars:
            if x.name == 'global/cnt:0':
                continue
            reg_loss += tf.nn.l2_loss(x)
        reg_loss *= self.regularization_lambda

        cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.actions,
                    logits=self.global_network.action_scores))

        self.total_loss = cross_entropy_loss + reg_loss

        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_op = optimizer.minimize(self.total_loss)

        tf.summary.scalar('reduced_ce_loss', cross_entropy_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()

        git_commit = check_output(
                'cd %s && git rev-parse @' % project_root.DIR, shell=True)
        date_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_name = date_time + '-%s' % git_commit.strip()
        self.logdir = path.join(project_root.DIR, 'dagger', 'logs', log_name)
        make_sure_path_exists(self.logdir)
        self.summary_writer = tf.summary.FileWriter(self.logdir)

    def wait_on_workers(self):
        """ Update which workers are done or dead. Stale tokens will
        eventually be cleaned out.
        Returns the number of workers that finished their episode.
        """
        workers_ep_done = 0
        while workers_ep_done < len(self.worker_tasks):
            # Let the workers dequeue their start tokens
            time.sleep(0.5)

            # check in each queue for worker messages and update workers
            workers_done = []
            for idx in self.worker_tasks:
                worker_queue = self.sync_queues[idx]
                msg = self.sess.run(worker_queue.dequeue())

                if msg == Status.EP_DONE:
                    workers_ep_done += 1
                elif msg == Status.WORKER_DONE:
                    workers_done.append(idx)
                    self.sess.run(worker_queue.close())
                else:
                    self.sess.run(worker_queue.enqueue(msg))

            for worker in workers_done:
                self.worker_tasks.remove(worker)

        return workers_ep_done

    def run_one_train_step(self, batch_states, batch_actions):
        """ Runs one step of the training operator on the given data.
        At times will update Tensorboard and save a checkpointed model.
        Returns the total loss calculated.
        """

        summary = True if self.train_step % 10 == 0 else False

        ops_to_run = [self.train_op, self.total_loss]

        if summary:
            ops_to_run.append(self.summary_op)

        pi = self.global_network

        start_ts = curr_ts_ms()
        ret = self.sess.run(ops_to_run, feed_dict={
            pi.input: batch_states,
            self.actions: batch_actions,
            pi.state_in: self.init_state})

        elapsed = (curr_ts_ms() - start_ts) / 1000.0
        sys.stderr.write('train step %d: time %.2f\n' %
                         (self.train_step, elapsed))

        if summary:
            self.summary_writer.add_summary(ret[2], self.train_step)

        return ret[1]

    def train(self):
        """ Runs the training operator until the loss converges.
        """
        curr_iter = 0

        min_loss = float('inf')
        iters_since_min_loss = 0

        batch_size = min(len(self.aggregated_states), self.default_batch_size)
        num_batches = len(self.aggregated_states) / batch_size

        if batch_size != self.default_batch_size:
            self.init_state = self.global_network.zero_init_state(batch_size)
        else:
            self.init_state = self.default_init_state

        while True:
            curr_iter += 1

            mean_loss = 0.0
            max_loss = 0.0

            for batch_num in xrange(num_batches):
                self.train_step += 1

                start = batch_num * batch_size
                end = start + batch_size

                batch_states = self.aggregated_states[start:end]
                batch_actions = self.aggregated_actions[start:end]

                loss = self.run_one_train_step(batch_states, batch_actions)

                mean_loss += loss
                max_loss = max(loss, max_loss)

            mean_loss /= num_batches

            sys.stderr.write('--- iter %d: max loss %.4f, mean loss %.4f\n' %
                             (curr_iter, max_loss, mean_loss))

            if max_loss < min_loss - 0.001:
                min_loss = max_loss
                iters_since_min_loss = 0
            else:
                iters_since_min_loss += 1

            if curr_iter > 50:
                break

            if iters_since_min_loss >= max(0.2 * curr_iter, 10):
                break

        self.sess.run(self.global_network.add_one)

        # copy trained variables from GPU to CPU
        self.sess.run(self.sync_op)

        print 'DaggerLeader:global_network:cnt', self.sess.run(self.global_network.cnt)
        print 'DaggerLeader:global_network_cpu:cnt', self.sess.run(self.global_network_cpu.cnt)
        sys.stdout.flush()

    def run(self, debug=False):
        for curr_ep in xrange(self.max_eps):
            if debug:
                sys.stderr.write('[PSERVER EP %d]: waiting for workers %s\n' %
                                 (curr_ep, self.worker_tasks))

            workers_ep_done = self.wait_on_workers()

            # If workers had data, dequeue ALL the samples and train
            if workers_ep_done > 0:
                while True:
                    num_samples = self.sess.run(self.train_q.size())
                    if num_samples == 0:
                        break

                    data = self.sess.run(self.train_q.dequeue())
                    self.aggregated_states.append(data[0])
                    self.aggregated_actions.append(data[1])

                if debug:
                    sys.stderr.write('[PSERVER]: start training\n')

                self.train()
            else:
                if debug:
                    sys.stderr.write('[PSERVER]: quitting...\n')
                break

            # Save the network model for testing every so often
            if curr_ep == self.checkpoint:
                self.save_model(curr_ep)
                self.checkpoint += self.checkpoint_delta

            # After training, tell workers to start another episode
            for idx in self.worker_tasks:
                worker_queue = self.sync_queues[idx]
                self.sess.run(worker_queue.enqueue(Status.WORKER_START))


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env):
        # Distributed tensorflow and logging related
        self.cluster = cluster
        self.env = env
        self.task_idx = task_idx
        self.leader_device = '/job:ps/task:0'
        self.worker_device = '/job:worker/task:%d' % task_idx
        self.num_workers = cluster.num_tasks('worker')

        # Buffers and parameters required to train
        self.curr_ep = 0
        self.state_buf = []
        self.action_buf = []
        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt

        self.aug_state_dim = self.state_dim + self.action_cnt
        self.prev_action = self.action_cnt - 1

        self.expert = TrueDaggerExpert(env)
        # Must call env.set_sample_action() before env.rollout()
        env.set_sample_action(self.sample_action)

        # Set up Tensorflow for synchronization, training
        self.setup_tf_ops()
        self.sess = tf.Session(
            server.target, config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()
        self.sess.run(self.sync_q.enqueue(Status.WORKER_DONE))

    def setup_tf_ops(self):
        """ Sets up the shared Tensorflow operators and structures
        Refer to DaggerLeader for more information
        """

        # Set up the shared global network and local network.
        with tf.device(self.leader_device):
            with tf.variable_scope('global_cpu'):
                self.global_network_cpu = DaggerLSTM(
                    state_dim=self.aug_state_dim, action_cnt=self.action_cnt)

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = DaggerLSTM(
                    state_dim=self.aug_state_dim, action_cnt=self.action_cnt)

        self.init_state = self.local_network.zero_init_state(1)
        self.lstm_state = self.init_state

        # Build shared queues for training data and synchronization
        self.train_q = tf.FIFOQueue(
                self.num_workers, [tf.float32, tf.int32],
                shared_name='training_feed')

        self.sync_q = tf.FIFOQueue(3, [tf.int16],
                shared_name=('sync_q_%d' % self.task_idx))

        # Training data is [[aug_state]], [action]
        self.state_data = tf.placeholder(
                tf.float32, shape=(None, self.aug_state_dim))
        self.action_data = tf.placeholder(tf.int32, shape=(None))
        self.enqueue_train_op = self.train_q.enqueue(
                [self.state_data, self.action_data])

        # Sync local network to global network (CPU)
        local_vars = self.local_network.trainable_vars
        global_vars = self.global_network_cpu.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            local_vars, global_vars)])

    def sample_action(self, state):
        """ Given a state buffer in the past step, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """
        cwnd = state[self.state_dim - 1]
        expert_action = self.expert.sample_action(cwnd)

        # For decision-making, normalize.
        norm_state = normalize(state)

        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action

        # Fill in state_buf, action_buf
        self.state_buf.append(aug_state)
        self.action_buf.append(expert_action)

        # Always use the expert on the first episode to get our bearings.
        if self.curr_ep == 0:
            self.prev_action = expert_action
            return expert_action

        # Get probability of each action from the local network.
        pi = self.local_network
        feed_dict = {
            pi.input: [[aug_state]],
            pi.state_in: self.lstm_state,
        }
        ops_to_run = [pi.action_probs, pi.state_out]
        action_probs, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take and update current LSTM state
        # action = np.argmax(np.random.multinomial(1, action_probs[0][0] - 1e-5))
        action = np.argmax(action_probs[0][0])
        self.prev_action = action

        return action

    def rollout(self):
        """ Start an episode/flow with an empty dataset/environment. """
        self.state_buf = []
        self.action_buf = []
        self.prev_action = self.action_cnt - 1
        self.lstm_state = self.init_state

        self.env.reset()
        self.env.rollout()

    def run(self, debug=False):
        """Runs for max_ep episodes, each time sending data to the leader."""

        pi = self.local_network
        while True:
            if debug:
                sys.stderr.write('[WORKER %d Ep %d] Starting...\n' %
                                 (self.task_idx, self.curr_ep))

            # Reset local parameters to global
            self.sess.run(self.sync_op)

            print 'DaggerWorker:global_network_cpu:cnt', self.sess.run(self.global_network_cpu.cnt)
            print 'DaggerWorker:local_network:cnt', self.sess.run(self.local_network.cnt)
            sys.stdout.flush()

            # Start a single episode, populating state-action buffers.
            self.rollout()

            if debug:
                queue_size = self.sess.run(self.train_q.size())
                sys.stderr.write(
                    '[WORKER %d Ep %d]: enqueueing a sequence of data '
                    'into queue of size %d\n' %
                    (self.task_idx, self.curr_ep, queue_size))

            # Enqueue a sequence of data into the training queue.
            self.sess.run(self.enqueue_train_op, feed_dict={
                self.state_data: self.state_buf,
                self.action_data: self.action_buf})
            self.sess.run(self.sync_q.enqueue(Status.EP_DONE))

            if debug:
                queue_size = self.sess.run(self.train_q.size())
                sys.stderr.write(
                    '[WORKER %d Ep %d]: finished queueing data. '
                    'queue size now %d\n' %
                    (self.task_idx, self.curr_ep, queue_size))

            if debug:
                sys.stderr.write('[WORKER %d Ep %d]: waiting for server\n' %
                                 (self.task_idx, self.curr_ep))

            # Let the leader dequeue EP_DONE
            time.sleep(0.5)

            # Wait until pserver finishes training by blocking on sync_q
            # Only proceeds when it finds a message from the pserver.
            msg = self.sess.run(self.sync_q.dequeue())
            while (msg != Status.WORKER_START and msg != Status.PS_DONE):
                self.sess.run(self.sync_q.enqueue(msg))
                time.sleep(0.5)
                msg = self.sess.run(self.sync_q.dequeue())

            if msg == Status.PS_DONE:
                break

            self.curr_ep += 1
