# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Huawei Technologies
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


import collections
import ConfigParser
import time
import sys
import socket
import select
import struct
import datetime
import numpy as np
import project_root
from os import path
from helpers.helpers import (curr_ts_ms, apply_op, READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)
from multiprocessing import Process, Queue, Pipe



def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.

    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
            for idx, action in enumerate(action_list)}


class Ack():
    def __init__(self, ack_data):
        self.seq_num, self.send_ts, self.sent_bytes, self.delivered_time, self.delivered, self.ack_bytes = ack_data


class Sender(object):
    # RL exposed class/static variables
    max_cwnd = 25000  # packet of 1500B
    min_cwnd = 10

    max_rtt_normorlize = 300.0  # ms
    max_delay_normorlize = max_rtt_normorlize
    max_delivery_rate_normorlize = 1000.0  # Mbps
    max_send_rate_normorlize = max_delivery_rate_normorlize

    max_steps = 1000
    max_test_time = 30000

    cfg = ConfigParser.ConfigParser()
    cfg_path = path.join(project_root.DIR, 'config.ini')
    cfg.read(cfg_path)
    state_dim = int(cfg.get('global', 'state_dim'))

    action_mapping = format_actions(["/2.0", "/1.025", "+0.0", "*1.025", "*2.0"])
    action_cnt = len(action_mapping)

    pkt_size = 1500
    usable_size = 1392

    def __init__(self, port=0, train=False, debug=False):
        self.train = train
        self.debug = debug

        # UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SENDBUF, 32*1024)
        self.sock.bind(('0.0.0.0', port))
        sys.stderr.write('[sender] Listening on port %s\n' % self.sock.getsockname()[1])

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        self.indigo_header = 28
        self.indigo_payload = 'x' * (Sender.usable_size - self.indigo_header)
        self.indigo_length = self.indigo_header + len(self.indigo_payload)

        if self.debug:
            self.sampling_file = open(path.join(project_root.DIR, 'env', 'sampling_time'), 'w', 0)

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10
        self.step_len_ms = 10
        self.pre_ack = 0

        # state variables of Indigo
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')
        self.rtt_ewma = None
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

        self.last_check_seq = 0
        self.last_check_seq_1 = 0
        self.last_check_time = None
        self.last_send_time = None

        self.sent_queue = collections.deque()

        # preemptive params
        self.preemptive = False
        self.gain_flag = 0
        if self.preemptive == True:
            self.gain_cycle = 8
            self.gain_cycle_index = 0
            self.gain_flag = 0
            self.ec = None
            self.last_gain_time = None

        self.pacing = 1

        self.last_delivered = 0
        self.sampling_flag = 0
        self.loss_rate = 0
        self.loss_pkt = 0

        self.max_send_rate_ewma = 0.0
        self.min_send_rate_ewma = float('inf')
        self.min_delivery_rate_ewma = float('inf')
        self.max_delivery_rate_ewma = 0.0
        self.min_delay_ewma = float('inf')
        self.max_delay_ewma = 0.0

        self.step_start_ms = None
        self.running = True
        self.start_phase_time = 0
        self.start_phase_max = 12

        self.rtt_buf = []
        if self.train:
            self.step_cnt = 0
            self.ts_first = None
        else:
            self.test_start_time = None
            self.pc = None
            self.test_name = None

        self.send_queue_out, self.send_queue_in  = Pipe(False)
        self.send_process = Process(target=self.send_process_func)

    def cleanup(self):
        if self.debug and self.sampling_file:
            self.sampling_file.close()
        self.sock.close()
        self.send_queue_out.close()
        self.send_queue_in.close()
        self.send_process.terminate()

    def handshake(self):
        """Handshake with peer receiver. Must be called before run()."""

        while True:
            msg, addr = self.sock.recvfrom(1600)
            if msg == 'Hello from receiver' and self.peer_addr is None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break

        self.sock.setblocking(0)  # non-blocking UDP socket
        self.test_start_time = curr_ts_ms()
        self.send_process.start()

    def set_sample_action(self, sample_action):
        """Set the policy. Must be called before run()."""
        self.sample_action = sample_action

    def update_state_relative(self, ack):
        """ Update the state variables listed in __init__() """
        self.pre_ack = self.next_ack
        self.next_ack = max(self.next_ack, ack.seq_num + 1)
        curr_time_ms = curr_ts_ms()

        # Update RTT
        rtt = float(curr_time_ms - ack.send_ts)
        self.min_rtt = min(self.min_rtt, rtt)
        self.min_rtt = max(1, self.min_rtt)
        rtt = max(1, rtt)
        #print rtt, self.min_rtt

        if self.train:
            if self.ts_first is None:
                self.ts_first = curr_time_ms
        self.rtt_buf.append(rtt)

        if self.rtt_ewma is None:
            self.rtt_ewma = rtt
        else:
            self.rtt_ewma = 0.875 * self.rtt_ewma + 0.125 * rtt

        delay = rtt - self.min_rtt
        if self.delay_ewma is None:
            self.delay_ewma = delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay

        self.min_delay_ewma = min(self.min_delay_ewma, self.delay_ewma)
        self.max_delay_ewma = max(self.max_delay_ewma, self.delay_ewma)

        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) / max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma is None:
            self.delivery_rate_ewma = delivery_rate
        else:
            self.delivery_rate_ewma = (0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)

        self.min_delivery_rate_ewma = min(self.min_delivery_rate_ewma, self.delivery_rate_ewma)
        self.max_delivery_rate_ewma = max(self.max_delivery_rate_ewma, self.delivery_rate_ewma)

        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)

        if self.send_rate_ewma is None:
            self.send_rate_ewma = send_rate
        else:
            self.send_rate_ewma = 0.875 * self.send_rate_ewma + 0.125 * send_rate

        self.min_send_rate_ewma = min(self.min_send_rate_ewma, self.send_rate_ewma)
        self.max_send_rate_ewma = max(self.max_send_rate_ewma, self.send_rate_ewma)

        # ##############################-BEGIN---Normorlized send/delivery rate min_ewma & max_ewma ################################
        self.min_send_rate_ewma_normorlized = self.min_send_rate_ewma / Sender.max_send_rate_normorlize
        self.max_send_rate_ewma_normorlized = self.max_send_rate_ewma / Sender.max_send_rate_normorlize
        self.min_delivery_rate_ewma_normorlized = self.min_delivery_rate_ewma / Sender.max_delivery_rate_normorlize
        self.max_delivery_rate_ewma_normorlized = self.max_delivery_rate_ewma / Sender.max_delivery_rate_normorlize
        # ##############################-BEGIN---Normorlized min_ewma, max_ewma ####################################################

        # ############################## BEGIN---Relative value (Normorlized): (x-min_ewma)/(max_ewma-min_ewma) ####################
        if self.max_delay_ewma-self.min_delay_ewma == 0:
            self.delay_rel = 0
        else:
            self.delay_rel = (self.delay_ewma-self.min_delay_ewma) / (self.max_delay_ewma-self.min_delay_ewma)

        if self.max_delivery_rate_ewma-self.min_delivery_rate_ewma == 0:
            self.deliver_rel = 0
        else:
            self.deliver_rel = (self.delivery_rate_ewma - self.min_delivery_rate_ewma) / (self.max_delivery_rate_ewma - self.min_delivery_rate_ewma)

        if self.max_send_rate_ewma-self.min_send_rate_ewma == 0:
            self.send_rel = 0
        else:
            self.send_rel = (self.send_rate_ewma-self.min_send_rate_ewma) / (self.max_send_rate_ewma-self.min_send_rate_ewma)
        # ############################## END---Relative value: (x-min_ewma)/(max_ewma-min_ewma) ###################################

        # ################################## BEGIN---Absolute value (Normorlized): x/MAX ##########################################
        self.rtt_abs = self.rtt_ewma / Sender.max_rtt_normorlize
        self.delay_abs = self.delay_ewma / self.min_rtt
        self.send_abs = self.send_rate_ewma / Sender.max_send_rate_normorlize
        self.delivery_abs = self.delivery_rate_ewma / Sender.max_delivery_rate_normorlize
        # ################################## END---Absolute value (Normorlized): x/MAX ############################################

    def take_action(self, action_idx):
        # old_cwnd = self.cwnd
        op, val = self.action_mapping[action_idx]
        self.cwnd = apply_op(op, self.cwnd, val)
        self.cwnd = max(Sender.min_cwnd, self.cwnd)
        self.cwnd = min(Sender.max_cwnd, self.cwnd)

        #print self.cwnd

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def msend(self, num):
        if num == 0:
            return

        data_array = []
        for i in xrange(num):
            send_ts = curr_ts_ms()
            data = (self.seq_num, send_ts, self.sent_bytes, self.delivered_time, self.delivered)

            data_array.append(data)

            self.seq_num += 1
            self.sent_bytes += self.indigo_length
            self.sent_queue.append((send_ts, self.sent_bytes))
            self.last_send_time = send_ts

        self.send_queue_in.send(data_array)

    def send(self):
        send_ts = curr_ts_ms()
        data = (self.seq_num, send_ts, self.sent_bytes, self.delivered_time, self.delivered)

        self.send_queue_in.send(data)

        self.seq_num += 1
        self.sent_bytes += self.indigo_length
        self.sent_queue.append((send_ts, self.sent_bytes))
        self.last_send_time = send_ts

        return 0

    def cal_loss_rate(self, ack):
        # sent_queue = self.sent_queue

        queue_len = len(self.sent_queue)
        if queue_len == 0:
            return 0

        send_ts = ack.send_ts
        sent_bytes = -1
        remove_num = 0
        for t, s in self.sent_queue:
            remove_num = remove_num + 1
            if t == send_ts:
                sent_bytes = s
                break
        # Do not fine the ts
        if sent_bytes == -1:
            return 0

        t, pre_sent_bytes = self.sent_queue[0]
        if (sent_bytes - pre_sent_bytes) == 0:
            return 0

        loss_rate = 1 - float(1.0*(self.delivered-self.last_delivered) / (sent_bytes - pre_sent_bytes))
        self.loss_pkt = ((sent_bytes - pre_sent_bytes) - (self.delivered-self.last_delivered)) / self.indigo_length

        self.last_delivered = self.delivered

        for i in xrange(remove_num-1):
            self.sent_queue.popleft()

        return max(0, loss_rate)

    def check_update(self, cur_time):
        if self.train:
            if cur_time - self.step_start_ms > self.step_len_ms:
                return True
        else:
            if (self.start_phase_time < self.start_phase_max and cur_time - self.step_start_ms > max(self.min_rtt, self.step_len_ms)) or (self.start_phase_time >= self.start_phase_max and cur_time - self.step_start_ms > max(self.min_rtt/4.0, self.step_len_ms)):
                if self.start_phase_time < self.start_phase_max:
                    self.start_phase_time = self.start_phase_time + 1
                return True

        return False

    def recv(self):
        try:
            unpaced_data, addr = self.sock.recvfrom(1600)
        except socket.error:
            return 0

        if addr != self.peer_addr or len(unpaced_data) < 28:
            return 0

        ack = Ack(struct.unpack('!iiqiqi', unpaced_data))
        # print ack.seq_num, ack.send_ts, ack.sent_bytes, ack.delivered_time, ack.delivered, ack.ack_bytes

        self.update_state_relative(ack)

        if self.step_start_ms is None:
            self.step_start_ms = curr_ts_ms()

        # At each step end, feed the state:
        cur_time = curr_ts_ms()
        if self.check_update(cur_time):
        #if cur_time - self.step_start_ms > self.step_len_ms: # step's end
            self.loss_rate = self.cal_loss_rate(ack)

            state0 = [self.delay_ewma,
                      self.delivery_rate_ewma,
                      self.send_rate_ewma,
                      self.cwnd/Sender.max_cwnd]
            state1 = [self.delay_rel,
                      self.deliver_rel,
                      self.send_rel,
                      self.cwnd/Sender.max_cwnd]
            state2 = [self.delay_rel,
                      self.delivery_abs,
                      self.loss_rate,
                      1.0*self.cwnd/Sender.max_cwnd]
            state5 = [self.rtt_abs, self.delay_abs, self.send_abs, self.delivery_abs, self.max_send_rate_ewma_normorlized,
                      self.max_delivery_rate_ewma_normorlized, self.loss_rate, 1.0*self.cwnd/Sender.max_cwnd]
            state6 = [self.rtt_abs, self.delay_abs, self.send_abs,
                      self.delivery_abs, self.loss_rate, 1.0*self.cwnd/Sender.max_cwnd]
            state7 = [self.rtt_abs, self.delay_abs, self.send_abs,
                      self.delivery_abs, 1.0*self.cwnd/Sender.max_cwnd]

            state8 = [self.rtt_abs, self.delay_abs, self.send_abs,
                      self.delivery_abs, self.loss_rate, self.gain_flag, 1.0*self.cwnd/Sender.max_cwnd] # preemptive state

            state_dict = {'state0': state0,
                          'state1': state1,
                          'state2': state2,
                          'state5': state5,
                          'state6': state6,
                          'state7': state7,
                          'state8': state8}

            selected_state = state_dict[Sender.cfg.get('global', 'state')]
            #print selected_state
            # time how long it takes to get an action from the NN
            if self.debug:
                start_sample = time.time()
            if not self.train:
                invoke_start_time = datetime.datetime.now()

            action = self.sample_action(selected_state)

            if action == -1:
                return -1

            if self.debug :
                self.sampling_file.write('%.2f ms\n' % (
                    (time.time() - start_sample) * 1000))

            self.take_action(action)

            self.delay_ewma = None
            self.delivery_rate_ewma = None
            self.send_rate_ewma = None

            self.step_start_ms = curr_ts_ms()

            if self.train:
                self.step_cnt += 1
                if self.step_cnt >= Sender.max_steps:
                    self.step_cnt = 0
                    self.running = False

                    self.compute_performance()
            else:
                if self.pc is not None:
                    self.pc.upload_cwnd(self.cwnd)
                    #print self.cwnd
                if curr_ts_ms() - self.test_start_time >= Sender.max_test_time:
                    if self.test_name is not None:
                        self.test_start_time = 0
                        self.performance_for_test()
                        time.sleep(0.2)  # wait for all the packeted arriving at receiver
                        self.running = False
                    else:
                        self.print_performance()
                        self.running = False
        return 1

    def set_perf_client(self, pc):
        self.pc = pc

    def set_expert_client(self, ec):
        self.ec = ec

    def get_recv_num(self):
        recv_num = self.cwnd / (Sender.max_cwnd/50)
        recv_num = min(50, recv_num)
        recv_num = max(1, recv_num)
        return recv_num

    def run(self):
        TIMEOUT = 1000  # ms

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS

        if self.last_check_time is None:
            self.last_check_time = curr_ts_ms()

        if self.last_send_time is None:
            self.last_send_time = curr_ts_ms()

        if self.preemptive == True:
            self.last_gain_time = curr_ts_ms()

        pre_time = datetime.datetime.now()
        borrowed_pkt = 0.0
        interval = 500.0

        while self.running:
            # FIXING BUG: Some time, it will not send pkt. check whether it sends out pkt in 5s, if no, return
            current_time = curr_ts_ms()
            if current_time - self.last_check_time > 5000:
                if self.last_check_seq == self.seq_num:
                    return -1
                else:
                    self.last_check_time = current_time
                    self.last_check_seq = self.seq_num

            if self.rtt_ewma is None:
                rtt = 100
            else:
                rtt = self.rtt_ewma

            if current_time - self.last_send_time > rtt:
                self.msend(1)

            if self.preemptive == True:
                if current_time - self.last_gain_time > rtt:
                    self.gain_cycle_index += 1
                    if self.gain_cycle_index % self.gain_cycle == 0:
                        self.gain_flag = 1
                        self.gain_cycle_index = 0
                    else:
                        self.gain_flag = 0
                    if self.train:
                        self.ec.gain_flag = self.gain_flag
                    self.last_gain_time = current_time

            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            events = self.poller.poll(TIMEOUT)
            if not events:  # timed out
                self.msend(1)

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    i = 0
                    while self.running and i < self.get_recv_num():
                        ret = self.recv()
                        if ret == 1:
                            i = i + 1
                        elif ret == 0:
                            break
                        elif ret == -1:  # expert server error
                            return -1
                    #res = self.recv()
                if flag & WRITE_FLAGS:
                    if self.window_is_open():

                        if self.pacing == 1:
                            if self.rtt_ewma is None:
                                n = 0.5
                            else:
                                n = self.cwnd / self.min_rtt


                            _now = datetime.datetime.now()
                            if ((_now - pre_time).microseconds >= interval):
                                n = (_now - pre_time).microseconds / interval * (n*(interval/1000.0))
                                pre_time = _now

                                borrowed_pkt += n - int(n)
                                if borrowed_pkt >= 1:
                                    n += int(borrowed_pkt)
                                    borrowed_pkt = borrowed_pkt - int(borrowed_pkt)

                                num = 0
                                n = int(n)

                                availbable_cwnd = int(self.cwnd) - (self.seq_num - self.next_ack)

                                c = min(availbable_cwnd, n)

                                # while num < c:
                                #     if self.send() == 0:
                                #         num = num + 1
                                #     else:
                                #         break
                                #borrowed_pkt += c - num
                                self.msend(c)
                        else:
                            cwnd = int(self.cwnd) - (self.seq_num - self.next_ack)
                            self.msend(cwnd)

        return 0

    # def send_process_func(self):
    #     while True:
    #         self.send_queue_out.poll()
    #         seq_num, send_ts, sent_bytes, delivered_time, delivered = self.send_queue_out.recv()
    #         data = (seq_num, send_ts, sent_bytes, delivered_time, delivered)
    #         packed_data = struct.pack('!iiqiq', *data)
    #         ret = -1
    #         while ret == -1:
    #             ret = self.sock.sendto(packed_data + self.indigo_payload, self.peer_addr)

    def send_process_func(self):
        while True:
            #self.send_queue_out.poll()
            data_array = self.send_queue_out.recv()

            for seq_num, send_ts, sent_bytes, delivered_time, delivered in data_array:
                packed_data = struct.pack('!iiqiq', seq_num, send_ts, sent_bytes, delivered_time, delivered)
                ret = -1
                while ret == -1:
                    try:
                        ret = self.sock.sendto(packed_data + self.indigo_payload, self.peer_addr)
                    except socket.error:
                        ret = -1

    def compute_performance(self):
        duration = curr_ts_ms() - self.ts_first
        tput = 0.008 * self.delivered / duration
        perc_delay = np.percentile(self.rtt_buf, 95)

        with open(path.join(project_root.DIR, 'env', 'perf'), 'a', 0) as perf:
            perf.write('%.2f %d\n' % (tput, perc_delay))

    def print_performance(self):
        duration = curr_ts_ms() - self.test_start_time
        tput = 0.008 * self.delivered / duration
        perc_delay = np.percentile(self.rtt_buf, 95)
        avg_delay = np.percentile(self.rtt_buf, 50)

        print ('thx: %.2f, delay: %d(95th) %d(50th), sent_bytes: %d\n' % (tput, perc_delay, avg_delay, self.sent_bytes))

    def set_test_name(self, name):
        self.test_name = name

    def performance_for_test(self):
        rtt_file = path.join(project_root.DIR, 'tests', 'rtt_loss', 'sender_rtt_'+self.test_name)
        file = open(rtt_file, 'w')
        file.write(str(self.sent_bytes)+'\n')
        for rtt in self.rtt_buf:
            file.write('%.2f\n' % rtt)
        file.close()
