import sys
import time
import project_root
import numpy as np
import tensorflow as tf
import datetime
import time
from tensorflow import contrib
from random import random
from os import path
from models import DaggerNetwork
from helpers.helpers import make_sure_path_exists, ewma


class STATUS:
    EP_DONE = 0
    WORKER_DONE = 1


class DaggerExpert(object):
    def __init__(self):
        pass

    def sample_action(self):
        return 1


class DaggerLeader(object):
    def __init__(self, cluster, server, worker_tasks, episode_max_steps):
        self.cluster = cluster
        self.server = server
        self.worker_tasks = worker_tasks
        self.num_workers = len(worker_tasks)
        self.episode_max_steps = episode_max_steps
        self.curr_ep = 0

        # Create the master network and training/sync queues
        with tf.variable_scope('global'):
            self.global_network = DaggerNetwork(
                    state_dim=self.state_dim, action_cnt=self.action_cnt)

        # Each element is [state], action
        # Start queue capacity at maximum possible training samples
        self.train_queue_capacity = episode_max_steps * self.num_workers
        self.train_q = tf.RandomShuffleQueue(
                self.train_queue_capacity, 0,
                [tf.float32, tf.int16],
                shapes=[[self.state_dim],[]],
                shared_name='training_feed')

        # Elements are worker task index to token message
        # Sync queue should overestimate due to some possible race conditions
        self.sync_q_capacity = self.num_workers * 2
        self.sync_q = tf.FIFOQueue(
                self.sync_q_capacity, [tf.int16, tf.int16],
                shared_name="sync_queue")

        # Create session
        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

    def run(self):
        while True:
            workers_ep_done = 0

            sys.stderr.write("PSERVER ep %d: waiting for workers to finish" % self.curr_ep)

            # Keep watching the sync queue for worker statuses
            while workers_ep_done < len(self.worker_tasks):
                token = self.sess.run(self.sync_q.dequeue())

                # Update the set of workers and the queue to reflect
                # which workers are done or dead.
                # Stale tokens will eventually be cleaned out
                if token[0] not in self.worker_tasks:
                    pass
                elif token[1] == STATUS.EP_DONE:
                    workers_ep_done += 1
                elif token[1] == STATUS.WORKER_DONE:
                    self.worker_tasks.remove(token[0])
                else:
                    self.sess.run(self.sync_q.enqueue(token))

            # Now dequeue all the examples from the queue and train
            # TODO Set up expert, summaries, and models training
            if workers_ep_done > 0:
                num_examples = workers_ep_done * self.episode_max_steps
                sys.stderr.write("PSERVER ep %d: reached dequeuing/training step" % self.curr_ep)
            else:
                break

            # After training, tell workers to start another episode
            worker_start_msgs = [
                    [idx for idx in self.worker_tasks],
                    [STATUS.WORKER_START] * len(self.worker_tasks
            ]
            self.sess.run(self.sync_q.enqueue_many(worker_start_msgs)

            self.curr_ep += 1


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env, dataset_size):
        # Distributed tensorflow related
        self.cluster = cluster
        self.env = env
        self.task_idx = task_idx
        self.is_chief = (task_idx == 0)
        self.worker_device = '/job:worker/task:%d' % task_idx
        self.num_workers = cluster.num_tasks('worker')
        self.time_file = open('/tmp/sample_action_time', 'w')

        # Buffers and parameters required to train
        self.state_buf = []
        self.action_buf = []
        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt

        # Counters and hyperparameters
        self.curr_ep = 0
        self.max_ep = 2000
        self.check_point = 1000
        self.train_queue_capacity = dataset_size
        self.ewma_window = 3    # alpha = 2 / (window+1)
        self.use_expert_prob = 0.75
        self.expert = DaggerExpert(self.ewma_window)

        # Must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # Setup local tensorflow ops and classifier
        self.setup_tf_ops()

        # Create session
        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()
        self.sess.run(self.sync_q.enqueue([self.task_idx, STATUS.WORKER_DONE]))

    def setup_tf_ops(self):
        """ Sets up the shared global neural network, global ep,
        and training and syncing queue and various ops.
        """
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                cluster=self.cluster)):

            with tf.variable_scope('global'):
                self.global_network = DaggerNetwork(
                        state_dim=self.state_dim, action_cnt=self.action_cnt)

        # Build shared queues for training data and synchronization
        # Refer to DaggerLeader for more information
        with tf.device("/job:ps/task:0")
            self.train_q = tf.RandomShuffleQueue(
                    self.train_queue_capacity, 0,
                    [tf.float32, tf.int16],
                    shapes=[[self.state_dim],[]],
                    shared_name='training_feed')

            self.sync_workers_q = tf.FIFOQueue(
                    self.num_workers * 2, [tf.int16, tf.int16],
                    shared_name="sync_queue")

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = DaggerNetwork(
                        state_dim=self.state_dim, action_cnt=self.action_cnt)

            # Training data is two lists of states and actions.
            self.states_data = tf.placeholder(tf.float32, shape=(2, None))
            self.action_data = tf.placeholder(tf.int16, shape=(None))
            self.enqueue_train_op = self.train_q.enqueue_many(
                    [self.states_data, self.actions_data])

        # Sync local network to global network
        local_vars = self.local_network.trainable_vars
        global_vars = self.global_network.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            local_vars, global_vars)])

    def sample_action(self, ep_state_buf):
        """ Given a buffer of states in the past episode, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """

        # ravel() is a faster flatten()
        flat_ep_state_buf = np.asarray(ep_state_buf, dtype=np.float32).ravel()
        ewma_delay = ewma(flat_ep_state_buf, self.emwa_window)
        expert_action = self.expert.sample_action(ewma_delay)

        self.state_buf.extend([[ewma_delay]])
        self.action_buf.append(expert_action)

        # Exponentially decaying probability to actually use the expert action
        if random() < (self.use_expert_prob ** self.curr_ep):
            return expert_action

        # Get probability of each action from the local network.
        start_time = time.time()
        pi = self.local_network
        action_probs = self.sess.run(pi.action_probs,
                                     feed_dict={pi.states: [[ewma_delay]]})
        elapsed_time = time.time() - start_time
        self.time_file.write('sample_action: %s sec.\n' % elapsed_time)

        # Choose an action to take
        action = np.argmax(np.random.multinomial(1, action_probs[0][0] - 1e-5))
        return action

    def rollout(self):
        """ Start an episode/flow with an empty dataset/environment. """
        self.state_buf = []
        self.action_buf = []
        self.env.reset()
        self.env.rollout()

    def run(self):
        """Runs for max_ep episodes, each time sending data to the leader."""

        pi = self.local_network

        while self.curr_ep < self.max_eps:
            sys.stderr.write('Current global ep: %d\n' % self.curr_ep)

            # Reset local parameters to global
            self.sess.run(self.sync_op)
            # Start a single episode, populating state-action buffers.
            self.rollout()

            # Enqueue all of the state/action pairs into the training queue.
            self.sess.run(self.enqueue_train_op, feed_dict={
                self.states_data: self.state_buf,
                self.action_data: self.action_buf})
            self.sess.run(self.sync_q.enqueue([self.task_idx, STATUS.EP_DONE]))

            # Wait until pserver finishes training by blocking on sync_q
            # Only proceeds when it finds its own message to.
            token = self.sess.run(self.sync_q.dequeue())
            while token[0] != self.task_idx:
                self.sess.run(self.sync_q.enqueue(token))
                token = self.sess.run(self.sync_q.dequeue())

            self.curr_ep += 1
