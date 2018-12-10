#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan
# Copyright 2018 Wei Wang, Yiyang Shao (Huawei Technologies)
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
import select
import socket
import argparse
from os import path
import numpy as np
import tensorflow as tf
from models import DaggerLSTM
from multiprocessing import Process, Pipe

from policy import Policy
from message import Message
import context
from helpers.utils import (READ_FLAGS, WRITE_FLAGS, ERR_FLAGS,
                           READ_ERR_FLAGS, ALL_FLAGS, timestamp_ms, one_hot)


class Sender(object):
    def __init__(self, ip, port):
        self.peer_addr = (ip, port)

        # non-blocking UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setblocking(0)  # set socket to non-blocking

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        # sequence numbers
        self.seq_num = 0
        self.next_ack = 0

        # check sender health
        self.health_check_ms = None
        self.pre_checked_seq_num = 0
        self.pre_checked_next_ack = 0

        # congestion control policy
        self.policy = None

        # sender's max run time
        self.run_time = None
        self.start_time = None

        # dedicate process for sending pkt
        self.send_queue_out, self.send_queue_in = Pipe(False)
        self.send_process = Process(target=self.__send_process_func)

# private
    def __window_is_open(self):
        return self.seq_num - self.next_ack < int(self.policy.cwnd)

    def __send(self):
        msg = Message(self.seq_num, timestamp_ms(), self.policy.bytes_sent,
                      self.policy.ack_recv_ts, self.policy.bytes_acked)
        try:
            self.sock.sendto(msg.to_data(), self.peer_addr)
        except socket.error:
            sys.stderr.write('send error\n')
            return -1

        self.seq_num += 1

        # tell policy that a datagram was sent
        self.policy.data_sent()

        return 0

    def __msend(self, num):
        if num == 0:
            return
        data_array = []
        for i in xrange(num):
            data = (self.seq_num, timestamp_ms(), self.policy.bytes_sent, self.policy.ack_recv_ts, self.policy.bytes_acked)
            data_array.append(data)
            self.seq_num += 1
            self.policy.data_sent()
        self.send_queue_in.send(data_array)

    def __send_process_func(self):
        while True:
            data_array = self.send_queue_out.recv()
            for seq_num, send_ts, bytes_sent, ack_recv_ts, bytes_acked in data_array:
                msg = Message(seq_num, send_ts, bytes_sent, ack_recv_ts, bytes_acked).to_data()
                ret = -1
                while ret == -1:
                    try:
                        ret = self.sock.sendto(msg, self.peer_addr)
                    except socket.error:
                        ret = -1
                        pass

    def __pacing_send(self):
        c = self.policy.pacing_pkt_number(self.policy.cwnd - (self.seq_num - self.next_ack))
        self.__msend(c)

    def __check_sender_health(self):
        if self.health_check_ms is None:
            self.health_check_ms = timestamp_ms()

        if timestamp_ms() - self.health_check_ms > 10000:  # cool down for 10s
            self.health_check_ms = timestamp_ms()

            if self.pre_checked_seq_num == self.seq_num or self.pre_checked_next_ack == self.next_ack:
                self.pre_checked_seq_num = self.seq_num
                self.pre_checked_next_ack = self.next_ack
                return False

            self.pre_checked_seq_num = self.seq_num
            self.pre_checked_next_ack = self.next_ack
        return True

    def __recv(self):
        try:
            msg_str, addr = self.sock.recvfrom(1500)
        except socket.error:
            return -1
        if len(msg_str) < Message.header_size:
            return -1

        ack = Message.parse(msg_str)

        # update next ACK's sequence number to expect
        self.next_ack = max(self.next_ack, ack.seq_num + 1)

        # tell policy that an ack was received
        self.policy.ack_received(ack)

        return 0

    def __run_timeout(self):
        if (self.run_time == None or self.policy.train):
            return False
            
        if timestamp_ms() - self.start_time > self.run_time:
            return True
        else:
            return False

# public
    def cleanup(self):
        self.sock.close()
        self.send_queue_out.close()
        self.send_queue_in.close()
        self.send_process.terminate()

    def set_policy(self, policy):
        self.policy = policy

    def set_run_time(self,time):
        self.run_time = time # ms

    def run(self):
        if not self.policy:
            sys.exit('sender\'s policy has not been set')

        self.start_time = timestamp_ms()
        self.send_process.start()
        while not self.policy.stop_sender and not self.__run_timeout():
            if not self.__check_sender_health():
                sys.stderr.write('No send or recv packets for 10 senconds. Exited.\n')
                return -1

            if self.__window_is_open():
                self.poller.modify(self.sock, ALL_FLAGS)
            else:
                self.poller.modify(self.sock, READ_ERR_FLAGS)
            events = self.poller.poll(self.policy.timeout_ms())
            if not events:  # timed out; send one datagram to get rolling
                self.__msend(1)

            for fd, flag in events:
                if flag & ERR_FLAGS:
                    sys.exit('[sender] error returned from poller')

                if flag & READ_FLAGS:
                    while self.__recv() == 0:
                        pass

                if flag & WRITE_FLAGS:
                    if self.__window_is_open():
                        if self.policy.pacing:
                            self.__pacing_send()
                        else:
                            num = int(self.policy.cwnd) - (self.seq_num - self.next_ack)
                            self.__msend(num)


class LSTMExecuter(object):
    # load model and make inference for standlone run
    def __init__(self, state_dim, action_cnt, restore_vars):
        self.aug_state_dim = state_dim + action_cnt
        self.action_cnt = action_cnt
        self.prev_action = action_cnt - 1

        with tf.variable_scope('global'):
            self.model = DaggerLSTM(
                state_dim=self.aug_state_dim, action_cnt=action_cnt)

        self.lstm_state = self.model.zero_init_state(1)

        self.sess = tf.Session()

        # restore saved variables
        saver = tf.train.Saver(self.model.trainable_vars)
        saver.restore(self.sess, restore_vars)

        # init the remaining vars, especially those created by optimizer
        uninit_vars = set(tf.global_variables())
        uninit_vars -= set(self.model.trainable_vars)
        self.sess.run(tf.variables_initializer(uninit_vars))

    def save_ckpt_model(self):
        # save the model in ckpt format
        cpp_model_path = path.join(context.base_dir, 'dagger', 'model_cpp')
        tf.train.write_graph(self.sess.graph_def, cpp_model_path, "cpp_model.pbtxt", as_text=True)
        checkpoint_path = path.join(context.base_dir, 'dagger', 'model_cpp', 'cpp_model.ckpt')
        self.model.saver.save(self.sess, checkpoint_path)
        print "export cpp model complete"

    def sample_action(self, state):
        # norm_state = normalize(state)
        norm_state = state

        one_hot_action = one_hot(self.prev_action, self.action_cnt)
        aug_state = norm_state + one_hot_action

        # Get probability of each action from the local network.
        pi = self.model
        feed_dict = {
            pi.input: [[aug_state]],
            pi.state_in: self.lstm_state,
        }

        ops_to_run = [pi.action_probs, pi.state_out]
        action_probs, self.lstm_state = self.sess.run(ops_to_run, feed_dict)

        # Choose an action to take
        action = np.argmax(action_probs[0][0])
        self.prev_action = action

        # action = np.argmax(np.random.multinomial(1, action_probs[0] - 1e-5))
        # temperature = 1.0
        # temp_probs = softmax(action_probs[0] / temperature)
        # action = np.argmax(np.random.multinomial(1, temp_probs - 1e-5))
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip')
    parser.add_argument('port', type=int)
    parser.add_argument('model', action='store')
    args = parser.parse_args()

    sender = None
    try:
        # dummy policy
        #policy = Policy(False)
        #policy.set_sample_action(lambda state: 2)

        # normal policy
        policy = Policy(False)
        lstm = LSTMExecuter(state_dim=Policy.state_dim,
                             action_cnt=Policy.action_cnt,
                             restore_vars=args.model)
        policy.set_sample_action(lstm.sample_action)

        sender = Sender(args.ip, args.port)
        sender.set_policy(policy)
        sender.run()
    except KeyboardInterrupt:
        sys.stderr.write('[sender] stopped\n')
    finally:
        if sender:
            sender.cleanup()


if __name__ == '__main__':
    main()
