import sys
import time
import project_root
import numpy as np
import tensorflow as tf
import datetime
import time
from tensorflow import contrib
from os import path
from models import DaggerNetwork
from experts import TrueDaggerExpert
from env.sender import Sender
from helpers.helpers import make_sure_path_exists, ewma


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
        self.curr_train_step = 0
        self.max_eps = 2000
        self.check_point = 1000
        self.num_batches = 10
        self.learn_rate = 1e-3
        self.loss_delta = 1e-1

        # Create the master network and training/sync queues
        with tf.variable_scope('global'):
            self.global_network = DaggerNetwork(
                    state_dim=Sender.state_dim, action_cnt=Sender.action_cnt)

        # Each element is [state], action
        # Queue capacity = maximum possible training samples
        self.train_queue_capacity = Sender.max_steps * self.num_workers
        self.train_q = tf.RandomShuffleQueue(
                self.train_queue_capacity, 0,
                [tf.float32, tf.int32],
                shapes=[[Sender.state_dim],[]],
                shared_name='training_feed')

        # Elements: [worker index, status message)
        # Extra space in the queue to take care of race conditions
        self.sync_q_capacity = self.num_workers * 2
        self.sync_q = tf.FIFOQueue(
                self.sync_q_capacity, [tf.int16, tf.int16],
                shared_name='sync_queue')

        self.setup_tf_ops(server)

    def cleanup(self):
        """ Sends messages to workers to stop. Tries hard to save the model """
        try:
            end_worker_msgs = [
                [idx for idx in self.worker_tasks],
                [Status.PS_DONE] * len(self.worker_tasks)]
            self.sess.run(self.sync_q.enqueue_many(end_worker_msgs))
        finally:
            self.save_model()

    def save_model(self, check_point=None):
        """ Takes care of saving/checkpointing the model. """
        if check_point is None:
            model_path = path.join(self.logdir, 'model')
        else:
            model_path = path.join(self.logdir, 'checkpoint-%d' % check_point)

        # save local parameters to worker-0
        saver = tf.train.Saver(self.global_network.trainable_vars)
        saver.save(self.sess, model_path)
        sys.stderr.write('\nModel saved to pserver at %s\n' % model_path)

    def setup_tf_ops(self, server):
        """ Sets up Tensorboard operators and tools, such as the optimizer,
        summary values, Tensorboard, and Session.
        """

        self.actions = tf.placeholder(tf.int32, [None])
        self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.actions, 
                    logits=self.global_network.action_scores))
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_step = optimizer.minimize(self.cross_entropy_loss)

        tf.summary.scalar('reduced_ce_loss', self.cross_entropy_loss)
        self.summary_op = tf.summary.merge_all()

        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

        date_time = datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.logdir = path.join(project_root.DIR, 'dagger', 'logs', date_time)
        make_sure_path_exists(self.logdir)
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

    def wait_on_workers(self):
        """ Update which workers are done or dead. Stale tokens will 
        eventually be cleaned out. 
        Returns the number of workers that finished their episode.
        """
        workers_ep_done = 0
        while workers_ep_done < len(self.worker_tasks):
            worker, msg = self.sess.run(self.sync_q.dequeue())

            if worker not in self.worker_tasks:
                pass
            elif msg == Status.EP_DONE:
                workers_ep_done += 1
            elif msg == Status.WORKER_DONE:
                self.worker_tasks.remove(worker)
            else:
                self.sess.run(self.sync_q.enqueue([worker, msg]))

        return workers_ep_done

    def train(self, num_examples, states, actions):
        """ Runs the training operator until the loss converges.
        Batches the state action pairs and repeatedly trains on
        those batches.
        """

        # In case number of batches > number of examples
        self.num_batches = min(num_examples, self.num_batches)
        batch_size = num_examples / self.num_batches
        prev_loss = float("inf")
        curr_loss = 1e10
        self.loss_delta = 1e-1
        batch_num = 0
        curr_ep_train_step = 0
        min_train_steps = self.num_batches * 2

        while ((prev_loss - curr_loss > self.loss_delta) or
               (curr_ep_train_step < min_train_steps)):

            ops_to_run = [self.train_step, self.cross_entropy_loss]

            if self.curr_train_step % self.num_batches == 0: 
                ops_to_run.append(self.summary_op)
                sys.stderr.write('On %d training step\n' % self.curr_train_step)

            start = batch_size * batch_num
            end = start + batch_size
            ret = self.sess.run(ops_to_run, feed_dict={
                self.global_network.states: states[start:end],
                self.actions: actions[start:end]}
            )

            if self.curr_train_step % self.num_batches == 0:
                summary = ret[2]
                self.train_writer.add_summary(summary, self.curr_train_step)

            prev_loss = curr_loss
            curr_loss = ret[1]
            self.curr_train_step += 1
            curr_ep_train_step += 1
            batch_num += 1
            batch_num = min(batch_num, self.num_batches-1)

            if self.curr_train_step == 1100:
                self.save_model(self.curr_train_step)

    def run(self, debug=False):
        for curr_ep in xrange(self.max_eps):

            if debug:
                sys.stderr.write('[PSERVER EP %d]: waiting for workers %s\n' %
                                (curr_ep, self.worker_tasks))

            workers_ep_done = self.wait_on_workers()
        
            # If workers had data, dequeue ALL the examples and train
            if workers_ep_done > 0:
                if debug:
                    sys.stderr.write('[PSERVER EP %d]: dequeueing\n' % curr_ep)

                num_examples = self.sess.run(self.train_q.size())
                states, actions = self.sess.run(
                        self.train_q.dequeue_many(num_examples))

                if debug:
                    sys.stderr.write('[PSERVER EP %d]: starting training\n' % curr_ep)

                self.train(num_examples, states, actions)
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
        self.is_chief = task_idx == 0
        if self.is_chief:
            self.time_file = open('/tmp/sample_action_time', 'w')

        # Distributed tensorflow and logging related
        self.cluster = cluster
        self.env = env
        self.task_idx = task_idx
        self.is_chief = (task_idx == 0)
        self.leader_device = '/job:ps/task:0'
        self.worker_device = '/job:worker/task:%d' % task_idx
        self.num_workers = cluster.num_tasks('worker')

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
        self.expert = TrueDaggerExpert(env)

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
        with tf.device(self.leader_device):
            self.train_q = tf.RandomShuffleQueue(
                    self.train_queue_capacity, 0,
                    [tf.float32, tf.int32],
                    shapes=[[self.state_dim],[]],
                    shared_name='training_feed')

            self.sync_q = tf.FIFOQueue(
                    self.num_workers * 2, [tf.int16, tf.int16],
                    shared_name='sync_queue')

        # Training data is [[state]], [action]
        self.states_data = tf.placeholder(tf.float32, 
                                          shape=(None, self.state_dim))
        self.action_data = tf.placeholder(tf.int32, shape=(None))
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
        start_time = time.time()

        # For ewma delay, only want first component, the one-way delay
        # For the cwnd, try only the most recent cwnd
        owd_buf = np.asarray([state[0] for state in step_state_buf])
        ewma_delay = ewma(owd_buf, self.ewma_window)
        last_cwnd = step_state_buf[-1][1]
        expert_action = self.expert.sample_action(ewma_delay, last_cwnd)

        self.state_buf.extend([[ewma_delay, last_cwnd]])
        self.action_buf.append(expert_action)

        # Exponentially decaying probability to actually use the expert action
        if np.random.random() < (self.use_expert_prob ** self.curr_ep):
            return expert_action

        # Get probability of each action from the local network.
        pi = self.local_network
        action_probs = self.sess.run(pi.action_probs,
                                     feed_dict={pi.states: [[ewma_delay,
                                                             last_cwnd]]})
        # Choose an action to take
        action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        
        if self.is_chief:
            elapsed_time = time.time() - start_time
            self.time_file.write('sample_action: %s sec\n' % elapsed_time)

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
            # Only proceeds when it finds its own message from the pserver.
            idx, msg = self.sess.run(self.sync_q.dequeue())
            while (idx != self.task_idx or
                    (msg != Status.WORKER_START and msg != Status.PS_DONE)):
                self.sess.run(self.sync_q.enqueue([idx, msg]))
                idx, msg = self.sess.run(self.sync_q.dequeue())

            if msg == Status.PS_DONE:
                break

            self.curr_ep += 1
