import sys
import time
import project_root
import numpy as np
import tensorflow as tf
import datetime
from tensorflow import contrib
from os import path
from models import DaggerNetwork
from experts import TrueDaggerExpert
from env.sender import Sender
from helpers.helpers import make_sure_path_exists, normalize


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
        self.curr_train_step = 0
        self.max_eps = 500
        self.checkpoint_delta = 20
        self.checkpoint = self.checkpoint_delta
        self.default_batch_size = 100
        self.learn_rate = 1e-3
        self.regularization_lambda = 1e-2

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

        # Keys: worker indices, values: Tensorflow messaging queues
        # Queue Elements: Status message)
        self.sync_queues = {}
        for idx in worker_tasks:
            queue_name = 'sync_q_%d' % idx
            self.sync_queues[idx] = tf.FIFOQueue(3, [tf.int16],
                                                 shared_name=queue_name)

        self.setup_tf_ops(server)

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

        # save local parameters to worker-0
        saver = tf.train.Saver(self.global_network.trainable_vars)
        saver.save(self.sess, model_path)
        sys.stderr.write('\nModel saved to param. server at %s\n' % model_path)

    def setup_tf_ops(self, server):
        """ Sets up Tensorboard operators and tools, such as the optimizer,
        summary values, Tensorboard, and Session.
        """

        self.actions = tf.placeholder(tf.int32, [None])

        reg_loss = 0.0
        for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            reg_loss += tf.nn.l2_loss(x)
        reg_loss *= self.regularization_lambda

        cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.actions,
                    logits=self.global_network.action_scores))

        self.total_loss = cross_entropy_loss + reg_loss

        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        self.train_step = optimizer.minimize(self.total_loss)

        tf.summary.scalar('reduced_ce_loss', cross_entropy_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', self.total_loss)
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

    def run_one_train_step(self, num_batches, batch_num, batch_size):
        """ Runs one step of the training operator on the given batch.
        At times will update Tensorboard and save a checkpointed model.
        Returns the total loss calculated.
        """
        ops_to_run = [self.train_step, self.total_loss]

        if self.curr_train_step % num_batches == 0:
            ops_to_run.append(self.summary_op)

        start = batch_size * batch_num
        end = start + batch_size

        ret = self.sess.run(ops_to_run, feed_dict={
            self.global_network.states: self.aggregated_states[start:end],
            self.actions: self.aggregated_actions[start:end]}
        )

        if self.curr_train_step % num_batches == 0:
            self.train_writer.add_summary(ret[2], self.curr_train_step)

        self.curr_train_step += 1

        return ret[1]

    def train(self):
        """ Runs the training operator until the loss converges.
        Batches the state action pairs and repeatedly trains on
        those batches.
        """
        dataset_size = len(self.aggregated_states)
        # In case the dataset is smaller than the batch size
        batch_size = min(dataset_size, self.default_batch_size)
        num_batches = dataset_size / batch_size

        min_loss = float("inf")
        iters_since_min_loss = 0
        curr_iter = 0

        # Stop condition: min # of steps and no smaller loss seen in a while
        while (iters_since_min_loss < max(0.25 * curr_iter, 5)):

            curr_loss = 0.0
            for i in xrange(num_batches):
                loss = self.run_one_train_step(num_batches, i, batch_size)
                curr_loss += loss
            curr_loss /= num_batches

            sys.stderr.write('[PSERVER] step %d: mean loss %.3f\n' %
                             (self.curr_train_step, curr_loss))

            if curr_loss < min_loss - 0.01:
                min_loss = curr_loss
                iters_since_min_loss = 0
            else:
                iters_since_min_loss += 1

            curr_iter += 1

    def run(self, debug=False):
        for curr_ep in xrange(self.max_eps):

            if debug:
                sys.stderr.write('[PSERVER EP %d]: waiting for workers %s\n' %
                                (curr_ep, self.worker_tasks))

            workers_ep_done = self.wait_on_workers()

            # If workers had data, dequeue ALL the examples and train
            if workers_ep_done > 0:

                num_examples = self.sess.run(self.train_q.size())
                states, actions = self.sess.run(
                        self.train_q.dequeue_many(num_examples))
                self.aggregated_states.extend(states)
                self.aggregated_actions.extend(actions)

                if debug:
                    sys.stderr.write('[PSERVER]: start training\n')

                self.train()

                if curr_ep == self.checkpoint:
                    self.save_model(curr_ep)
                    self.checkpoint += self.checkpoint_delta
            else:
                if debug:
                    sys.stderr.write('[PSERVER]: quitting...\n')
                break

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

        # Hyperparameters
        self.train_queue_capacity = Sender.max_steps * self.num_workers
        self.expert = TrueDaggerExpert(env)

        # Must call env.set_sample_action() before env.run()
        env.set_sample_action(self.sample_action)

        # Set up Tensorflow for synchronization, training
        self.setup_tf_ops()
        self.sess = tf.Session(server.target)
        self.sess.run(tf.global_variables_initializer())

    def cleanup(self):
        self.env.cleanup()
        self.sess.run(self.sync_q.enqueue(Status.WORKER_DONE))

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

            self.sync_q = tf.FIFOQueue(3, [tf.int16],
                    shared_name=('sync_q_%d' % self.task_idx))

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

    def sample_action(self, state):
        """ Given a state buffer in the past step, returns an action
        to perform.

        Appends to the state/action buffers the state and the
        "correct" action to take according to the expert.
        """
        step_cwnd = state[-1]
        expert_action = self.expert.sample_action(step_cwnd)

        # For decision-making, normalize.
        state = normalize(state)

        self.state_buf.extend([state])
        self.action_buf.append(expert_action)

        # Always use the expert on the first episode to get our bearings.
        if self.curr_ep == 0:
            return expert_action

        # Get probability of each action from the local network.
        pi = self.local_network
        action_probs = self.sess.run(pi.action_probs,
                                     feed_dict={pi.states: [state]})

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
            self.sess.run(self.sync_q.enqueue(Status.EP_DONE))

            if debug:
                queue_size = self.sess.run(self.train_q.size())
                sys.stderr.write(('[WORKER %d Ep %d]: finished queueing data. '
                                  'queue size now %d\n')
                                  % (self.task_idx, self.curr_ep, queue_size))

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
