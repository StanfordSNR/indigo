# TODO Set up expert, summaries, and models training
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
from env.sender import Sender
from helpers.helpers import make_sure_path_exists, ewma


class Status:
    EP_DONE = 0
    WORKER_DONE = 1
    WORKER_START = 2
    PS_DONE = 3


class DaggerExpert(object):
    """ Naive LEDBAT implementation """
    def __init__(self):
        self.base_delay = float("inf")
        self.target = 100
        self.gain = 1
        self.actions_index = zip(Sender.action_mapping,
                                 xrange(len(Sender.action_mapping)))

    def sample_action(self, ewma_delay, cwnd):
        self.base_delay = min(self.base_delay, ewma_delay)
        queuing_delay = ewma_delay - self.base_delay
        off_target = self.target - queuing_delay
        cwnd_inc = self.gain * off_target / cwnd

        # Gets the action closest to the actual increase to the cwnd
        action = min(self.actions_index, key=lambda x: abs(x[0]-cwnd_inc))
        return int(action[1])

class DaggerLeader(object):
    def __init__(self, cluster, server, worker_tasks):

        self.cluster = cluster
        self.server = server
        self.worker_tasks = worker_tasks
        self.num_workers = len(worker_tasks)
        self.episode_max_steps = Sender.max_steps
        self.max_eps = 2000

        # Create the master network and training/sync queues
        with tf.variable_scope('global'):
            self.global_network = DaggerNetwork(
                    state_dim=Sender.state_dim, action_cnt=Sender.action_cnt)

        # Each element is [state], action
        # Queue capacity = maximum possible training samples
        self.train_queue_capacity = self.episode_max_steps * self.num_workers
        self.train_q = tf.RandomShuffleQueue(
                self.train_queue_capacity, 0,
                [tf.float32, tf.int16],
                shapes=[[Sender.state_dim],[]],
                shared_name='training_feed')

        # Elements: [worker index, status message)
        # Extra space in the queue to take care of race conditions
        self.sync_q_capacity = self.num_workers * 2
        self.sync_q = tf.FIFOQueue(
                self.sync_q_capacity, [tf.int16, tf.int16],
                shared_name='sync_queue')

        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        end_worker_msgs = [
                [idx for idx in self.worker_tasks],
                [Status.PS_DONE] * len(self.worker_tasks)]
        self.sess.run(self.sync_q.enqueue_many(worker_start_msgs))

    def wait_on_workers(self):
        """ Update which workers are done or dead. Stale tokens will 
        eventually be cleaned out. 
        Returns the number of workers that finished their episode.
        """
        workers_ep_done = 0
        while workers_ep_done < len(self.worker_tasks):
            token = self.sess.run(self.sync_q.dequeue())

            if token[0] not in self.worker_tasks:
                pass
            elif token[1] == Status.EP_DONE:
                workers_ep_done += 1
            elif token[1] == Status.WORKER_DONE:
                self.worker_tasks.remove(token[0])
            else:
                self.sess.run(self.sync_q.enqueue(token))

        return workers_ep_done

    def run(self, debug=False):
        for curr_ep in xrange(self.max_eps):

            if debug:
                sys.stderr.write('[PSERVER EP %d]: waiting for workers %s\n' %
                                (curr_ep, self.worker_tasks))

            workers_ep_done = self.wait_on_workers()
        
            # If workers had data, dequeue ALL the examples and train
            if workers_ep_done > 0:
                if debug:
                    sys.stderr.write('[PSERVER EP %d]: dequeueing\n'
                                     % curr_ep)

                num_examples = workers_ep_done * self.episode_max_steps
                states, actions = self.sess.run(
                        self.train_q.dequeue_many(num_examples))

                if debug:
                    sys.stderr.write(('[PSERVER EP %d]: finished dequeueing\n'
                                      '     states: %s\n'
                                      '     actions: %s\n') 
                                      % (curr_ep, states, actions))

                # train, using mini batches from the states and actions
               

            else:
                if debug:
                    sys.stderr.write('[PSERVER ep %d]: quitting...\n' 
                                     % curr_ep)
                break

            # After training, tell workers to start another episode
            worker_start_msgs = [
                    [idx for idx in self.worker_tasks],
                    [Status.WORKER_START] * len(self.worker_tasks)]
            self.sess.run(self.sync_q.enqueue_many(worker_start_msgs))


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env):
        # Distributed tensorflow and logging related
        self.cluster = cluster
        self.env = env
        self.task_idx = task_idx
        self.is_chief = (task_idx == 0)
        self.worker_device = '/job:worker/task:%d' % task_idx
        self.num_workers = cluster.num_tasks('worker')
        self.time_file = open('/tmp/sample_action_time', 'w')

        # Buffers and parameters required to train
        self.curr_ep = 0
        self.state_buf = []
        self.action_buf = []
        self.state_dim = env.state_dim
        self.action_cnt = env.action_cnt

        # Hyperparameters
        self.train_queue_capacity = Sender.max_steps * self.num_workers
        self.ewma_window = 3    # alpha = 2 / (window+1)
        self.use_expert_prob = 0.75
        self.expert = DaggerExpert()

        # Must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # Set up Tensorflow for synchronization, training
        self.setup_tf_ops()
        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()
        self.sess.run(self.sync_q.enqueue([self.task_idx, Status.WORKER_DONE]))

    def setup_tf_ops(self):
        """ Sets up the shared Tensorflow operators and structures
        Refer to DaggerLeader for more information
        """

        # Set up the shared global network and local network.
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                cluster=self.cluster)):

            with tf.variable_scope('global'):
                self.global_network = DaggerNetwork(
                        state_dim=self.state_dim, action_cnt=self.action_cnt)

        with tf.device(self.worker_device):
            with tf.variable_scope('local'):
                self.local_network = DaggerNetwork(
                        state_dim=self.state_dim, action_cnt=self.action_cnt)

        # Build shared queues for training data and synchronization
        with tf.device('/job:ps/task:0'):
            self.train_q = tf.RandomShuffleQueue(
                    self.train_queue_capacity, 0,
                    [tf.float32, tf.int16],
                    shapes=[[self.state_dim],[]],
                    shared_name='training_feed')

            self.sync_q = tf.FIFOQueue(
                    self.num_workers * 2, [tf.int16, tf.int16],
                    shared_name='sync_queue')

        # Training data is [[state]], [action]
        self.states_data = tf.placeholder(tf.float32, 
                                          shape=(None, self.state_dim))
        self.action_data = tf.placeholder(tf.int16, shape=(None))
        self.enqueue_train_op = self.train_q.enqueue_many(
                [self.states_data, self.action_data])

        # Sync local network to global network
        local_vars = self.local_network.trainable_vars
        global_vars = self.global_network.trainable_vars
        self.sync_op = tf.group(*[v1.assign(v2) for v1, v2 in zip(
            local_vars, global_vars)])

    def sample_action(self, step_state_buf):
        """ Given a buffer of states in the past step, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """

        # For ewma delay, only want first component, the one-way delay
        # For the cwnd, try only the most recent cwnd
        owd_buf = np.asarray([state[0] for state in step_state_buf])
        ewma_delay = ewma(owd_buf, self.ewma_window)
        last_cwnd = step_state_buf[-1][1]
        expert_action = self.expert.sample_action(ewma_delay, last_cwnd)

        self.state_buf.extend([[ewma_delay, last_cwnd]])
        self.action_buf.append(expert_action)

        # Exponentially decaying probability to actually use the expert action
        if random() < (self.use_expert_prob ** self.curr_ep):
            return expert_action

        # Get probability of each action from the local network.
        start_time = time.time()
        pi = self.local_network
        action_probs = self.sess.run(pi.action_probs,
                                     feed_dict={pi.states: step_state_buf})
        elapsed_time = time.time() - start_time
        self.time_file.write('sample_action: %s sec.\n' % elapsed_time)

        # Choose an action to take
        action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        return action

    def rollout(self):
        """ Start an episode/flow with an empty dataset/environment. """
        self.state_buf = []
        self.action_buf = []
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
            # Start a single episode, populating state-action buffers.
            self.rollout()

            if debug:
                queue_size = self.sess.run(self.train_q.size())
                num_examples = len(self.action_buf)
                sys.stderr.write(('[WORKER %d Ep %d]: enqueueing %d examples '
                                  'into queue of size %d\n')
                                  % (self.task_idx, self.curr_ep,
                                     num_examples, queue_size))

            # Enqueue all of the state/action pairs into the training queue.
            self.sess.run(self.enqueue_train_op, feed_dict={
                self.states_data: self.state_buf,
                self.action_data: self.action_buf})
            self.sess.run(self.sync_q.enqueue([self.task_idx, Status.EP_DONE]))

            if debug:
                queue_size = self.sess.run(self.train_q.size())
                sys.stderr.write(('[WORKER %d Ep %d]: finished queueing data. '
                                  'queue size now %d\n')
                                  % (self.task_idx, self.curr_ep, queue_size))

            if debug:
                sys.stderr.write('[WORKER %d Ep %d]: waiting for server\n' %
                                (self.task_idx, self.curr_ep))

            # Wait until pserver finishes training by blocking on sync_q
            # Only proceeds when it finds its own message to.
            token = self.sess.run(self.sync_q.dequeue())
            while token[0] != self.task_idx:
                self.sess.run(self.sync_q.enqueue(token))
                token = self.sess.run(self.sync_q.dequeue())

            if token[1] == Status.PS_DONE:
                break

            self.curr_ep += 1
