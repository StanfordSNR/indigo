#!/usr/bin/env python

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


import ast
import ConfigParser
import datetime
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import project_root
import socket
import subprocess
import sys
import threading
from multiprocessing import Process, Queue
from env.sender import Sender

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.use('GTK')


class Device_QDISC():
    def __init__(self, dev_name):
        self.dev_name = dev_name
        self.sent_bytes = 0
        self.sent_pkt = 0
        self.dropped_pkt = 0
        self.record_time = datetime.datetime.now()

        # Emulator configuration
        self.BW_CAPACITY = 1000     # Mbps
        self.MAX_QUEUE_SIZE = 1000  # number
        self.LOSS_RATE = 0          # %
        self.MIN_RTT = 10           # ms

    def get_emulator_configuration(self):
        try:
            out_bytes = subprocess.check_output(
                ['tc', '-p', '-s', '-d', 'qdisc', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e
        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join(
            [x for x in out_slice if x.find('qdisc netem') != -1])

        if target_line.find('limit') != -1:
            self.MAX_QUEUE_SIZE = int(target_line.split(' ')[6])
        if target_line.find('delay') != -1:
            self.MIN_RTT = 2*float(target_line.split(' ')[8][:-2])
        if target_line.find('loss') != -1:
            self.LOSS_RATE = float(target_line.split(' ')[10][:-1])

        try:
            out_bytes = subprocess.check_output(
                ['tc', '-p', '-s', '-d', 'class', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e
        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join(
            [x for x in out_slice if x.find('class htb') != -1])

        _rate = target_line.split(' ')[11]
        if _rate[-4] == 'G':
            self.BW_CAPACITY = int(_rate[:-4])*1000
        elif _rate[-4] == 'M':
            self.BW_CAPACITY = int(_rate[:-4])
        else:
            self.BW_CAPACITY = int(_rate[:-4])/1000.0

        print 'Emulator Configuration: max queue size(pkt), loss rate(%), min RTT(ms), bandwidth capacity(Mbps)'
        self.MAX_QUEUE_SIZE = self.MAX_QUEUE_SIZE - self.BW_CAPACITY*(self.MIN_RTT/2)*1000/Sender.pkt_size/8
        print self.MAX_QUEUE_SIZE, self.LOSS_RATE, self.MIN_RTT, self.BW_CAPACITY

    def get_network_state(self, flag):
        _record_time = datetime.datetime.now()

        try:
            out_bytes = subprocess.check_output(
                ['tc', '-p', '-s', '-d', 'qdisc', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            raise e

        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')

        try:
            _sent_bytes = int(out_slice[1].split(' ')[2])
            _sent_pkt = int(out_slice[1].split(' ')[4])
            _dropped_pkt = int(out_slice[1].split(' ')[7][:-1])
            # _backlog_b  = int(out_slice[2].split(' ')[2][:-1])
            _backlog_p = int(out_slice[2].split(' ')[3][:-1])
            # IMPORTANT, _backlog_p includes the queues used for the delay of netem, so we should minus the queue size of one BDP
            _backlog_p_congestion = max(0, _backlog_p-self.BW_CAPACITY*(self.MIN_RTT/2)*1000/Sender.pkt_size/8)

            # print _sent_bytes,_sent_pkt,_dropped_pkt,_backlog_p
        except IndexError:
            return 0, 0, 0, 0

        if self.sent_bytes != 0:
            record_time = (_record_time - self.record_time).microseconds

            available_bw = self.BW_CAPACITY - 8 * (_sent_bytes - self.sent_bytes)/(1.0*record_time)
            queueing_factor = 1.0 * _backlog_p_congestion / self.MAX_QUEUE_SIZE
            congestion_loss = ((_sent_bytes - self.sent_bytes)/Sender.pkt_size + (_dropped_pkt-self.dropped_pkt)) - 1.0*(_sent_bytes - self.sent_bytes)/Sender.pkt_size/((100.0-self.LOSS_RATE)/100)
            non_congestion_loss = (_dropped_pkt-self.dropped_pkt) - congestion_loss
            # print time, available_bw, queueing_factor, non_congestion_loss, congestion_loss
        else:
            available_bw = 0
            queueing_factor = 0
            non_congestion_loss = 0
            congestion_loss = 0

        if flag == 1:
            self.sent_bytes = _sent_bytes
            self.sent_pkt = _sent_pkt
            self.dropped_pkt = _dropped_pkt
            self.record_time = _record_time

        return available_bw, queueing_factor, non_congestion_loss, congestion_loss, _backlog_p


class Device_IFCONFIG():
    def __init__(self, dev_name):
        self.dev_name = dev_name

        self.rx_packets = 0
        self.rx_bytes = 0
        self.record_time = datetime.datetime.now()

    def get_rx_rate(self, flag):
        try:
            out_bytes = subprocess.check_output(['ifconfig', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e
        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('RX packets') != -1])

        _rx_packets = int(target_line.split(':')[1].split(' ')[0])
        _record_time = datetime.datetime.now()
        if self.rx_packets != 0:
            rx_rate = 1.0*(_rx_packets - self.rx_packets) * (Sender.pkt_size) * 8 / (_record_time - self.record_time).microseconds
        else:
            rx_rate = 0

        if flag == 1:
            self.rx_packets = _rx_packets
            self.record_time = _record_time

        return rx_rate

    def get_rx_rate_bytes(self, flag):
        try:
            out_bytes = subprocess.check_output(['ifconfig', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e
        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('RX bytes') != -1])

        _rx_bytes = int(target_line.split(':')[1].split(' ')[0])
        _record_time = datetime.datetime.now()
        if self.rx_bytes != 0:
            rx_bytes = 1.0*(_rx_bytes - self.rx_bytes) * 8 / (_record_time - self.record_time).microseconds
        else:
            rx_bytes = 0

        if flag == 1:
            self.rx_bytes = _rx_bytes
            self.record_time = _record_time

        return rx_bytes


class Expert_Mininet():
    def __init__(self, dev_bottleneck, dev_traffics):

        self.dev_bottleneck = Device_QDISC(dev_bottleneck)
        self.dev_traffic_generator = Device_IFCONFIG(dev_traffics[0])
        self.dev_indigo_sender = Device_IFCONFIG(dev_traffics[1])
        self.flag = 1

    def init_configure(self):
        try:
            # time.sleep(2) #wait mininet to start up
            self.dev_bottleneck.get_emulator_configuration()
        except subprocess.CalledProcessError:
            sys.stderr.write('get emulator coniguration error Mininet Emulator does not start')
            return

    def best_cwnd_algorithm_BDP(self, available_bw_bn, tg_traffic, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn):
        # Current method: best_cwnd = available_bw * min_rtt

        best_send_rate = self.dev_bottleneck.BW_CAPACITY - (tg_traffic)  # Mbps
        if (best_send_rate) < 0:
            best_send_rate = 0
        best_cwnd = 1000 * best_send_rate * self.dev_bottleneck.MIN_RTT  # Mbps * ms = 10^6 b/s * 10^(-3) s = b
        best_cwnd = best_cwnd / (8 * Sender.pkt_size)
        return best_cwnd

    def best_cwnd_algorithm_x(self, available_bw_bn, tg_traffic, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn, queue_size_bn):
        # Current method: best_cwnd = available_bw * min_rtt
        #                 best_cwnd = best_cwnd + q * left_queue_size #  0 <= q <= 1, q is the aggressive factor

        cfg = ConfigParser.ConfigParser()
        cfg_path = os.path.join(project_root.DIR, 'config.ini')
        cfg.read(cfg_path)
        q = float(cfg.get('global', 'fri'))

        best_send_rate = self.dev_bottleneck.BW_CAPACITY - (tg_traffic)  # Mbps
        if (best_send_rate) < 0:
            best_send_rate = 0
        best_cwnd = 1000 * best_send_rate * self.dev_bottleneck.MIN_RTT  # Mbps * ms = 10^6 b/s * 10^(-3) s = b
        best_cwnd = best_cwnd / (8 * Sender.pkt_size)

        # left_queue_size = 1.0*self.dev_bottleneck.MAX_QUEUE_SIZE - queue_size_bn
        # left_queue_size = max(0, left_queue_size)
        best_cwnd_aggressive = best_cwnd + (self.dev_bottleneck.MAX_QUEUE_SIZE * q - self.dev_bottleneck.MAX_QUEUE_SIZE * queueing_factor_bn)
        # print "bdp best cwnd, best_cwnd aggressiive, non_congestion_loss_bn, congestion_loss_bn", best_cwnd, best_cwnd_aggressive, non_congestion_loss_bn, congestion_loss_bn
        return best_cwnd_aggressive

    def calculate_oracle(self):
        global indigo_cwnd, perf_queue
        '''
        TODO:
        Method 1: calculate the best sending rate and best_cwnd
        Method 2: give a score as the reward"
        '''
        flag = self.flag
        if self.flag == 1:
            self.flag = 0
        try:
            available_bw_bn, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn, queue_size_bn = self.dev_bottleneck.get_network_state(flag)
        except subprocess.CalledProcessError:
            sys.stderr.write('Mininet Emulator does not start')
            return
        try:
            tg_traffic = self.dev_traffic_generator.get_rx_rate_bytes(flag)
            sender_traffic = self.dev_indigo_sender.get_rx_rate(flag)
        except subprocess.CalledProcessError:
            sys.stderr.write('ifconfig error')
            return

        available_bw_bn = max(0, available_bw_bn)
        available_bw_bn = min(self.dev_bottleneck.BW_CAPACITY, available_bw_bn)
        tg_traffic = max(0, tg_traffic)
        tg_traffic = min(self.dev_bottleneck.BW_CAPACITY, tg_traffic)
        queueing_factor_bn = max(0.0, queueing_factor_bn)
        queueing_factor_bn = min(1.0, queueing_factor_bn)
        non_congestion_loss_bn = max(0, non_congestion_loss_bn)
        non_congestion_loss_bn = min(self.dev_bottleneck.MAX_QUEUE_SIZE, non_congestion_loss_bn)

        # We can implement differnet oracle to guide the traininng
        # best_cwnd = self.best_cwnd_algorithm_BDP(available_bw_bn, tg_traffic, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn)
        best_cwnd = self.best_cwnd_algorithm_x(available_bw_bn, tg_traffic, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn, queue_size_bn)
        best_cwnd = max(2, best_cwnd)

        if flag == 1:
            perf_queue.put((best_cwnd, indigo_cwnd))
            log_info = (self.dev_bottleneck.BW_CAPACITY, self.dev_bottleneck.MIN_RTT, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn, tg_traffic, sender_traffic, best_cwnd, indigo_cwnd)
            logging.info('{}, {}, {}, {}, {}, {}, {}, {}, {}'.format(*log_info))


def show_performance(pq, curr_env_name, curr_tp_name):
    best_cwnd_list = []
    best_cwnd_avg_list = []
    indigo_cwnd_list = []
    indigo_cwnd_avg_list = []
    draw_flag = 0

    # begin to draw
    plt.figure(figsize=(8, 6), dpi=80)
    plt.ion()

    while True:
        while not pq.empty():  # sync queue data
            best_cwnd, indigo_cwnd = pq.get(True)
            best_cwnd_list.append(best_cwnd)
            indigo_cwnd_list.append(indigo_cwnd)
            draw_flag += 1
            if draw_flag >= 10:
                best_cwnd_avg_list.append(np.mean(best_cwnd_list))
                indigo_cwnd_avg_list.append(np.mean(indigo_cwnd_list))
                best_cwnd_list = []
                indigo_cwnd_list = []
                draw_flag = 0

        plt.cla()
        plt.title('{} & tp_{}\nindigo cwnd v.s best cwnd'.format(curr_env_name, curr_tp_name))
        plt.grid(True)

        xlim_left = max(0, len(best_cwnd_avg_list)-100)
        y_best = best_cwnd_avg_list[-100:]
        y_indigo = indigo_cwnd_avg_list[-100:]
        x = range(xlim_left, xlim_left+len(y_best))

        plt.xlabel('iter times')
        plt.xlim(xlim_left, xlim_left + 120)
        plt.xticks(np.linspace(xlim_left, xlim_left + 120, 13, endpoint=True))

        plt.ylabel('cwnd value')
        plt.autoscale(enable=True, axis='y', tight=False)
        # plt.ylim(0, 1200)
        # plt.yticks(np.linspace(0, 1200, 7, endpoint=True))

        plt.plot(x, y_best, "b--", linewidth=2.0, label="best")
        plt.plot(x, y_indigo, "g-", linewidth=2.0, label="indigo")

        plt.legend(loc="upper left", shadow=True)
        plt.pause(0.1)
    plt.ioff()
    plt.show()


def get_mininet_env_param():
    total_tp_set = []
    total_env_name_set = []

    cfg = ConfigParser.ConfigParser()
    cfg_path = os.path.join(project_root.DIR, 'config.ini')
    cfg.read(cfg_path)

    test_env = cfg.options('test_env')
    for opt in test_env:
        env_param, tp_set_param = ast.literal_eval(cfg.get('test_env', opt))
        total_tp_set.append(ast.literal_eval(cfg.get('global', tp_set_param)))
        total_env_name_set.append(env_param)

    return total_tp_set, total_env_name_set


total_tp_set, total_env_name_set = get_mininet_env_param()
curr_env_name = total_env_name_set[int(sys.argv[2])]
curr_tp_name = total_tp_set[int(sys.argv[3])][int(sys.argv[4])]

perf_queue = Queue()
perf_process = Process(target=show_performance, args=(perf_queue, curr_env_name, curr_tp_name))

cwnd = 0
running = 0
indigo_cwnd = 0.0
exp = Expert_Mininet('s1-eth1', ['s1-eth2', 'veth2'])


def thread_fun():
    global cwnd, running
    while (running == 0):
        pass
    exp.init_configure()
    while True:
        exp.calculate_oracle()


if __name__ == '__main__':
    log_file_name = '{}_tp_{}_performance.log'.format(curr_env_name, curr_tp_name)
    log_file_path = os.path.join(project_root.DIR, 'tests', 'perf_log', log_file_name)
    logging.basicConfig(level=logging.INFO, format='%(message)s', filename=log_file_path)
    logging.info('bandwidth, min_rtt, queueing_factor, non_congestion_loss, congestion_loss_bn, tg_traffic, sender_traffic, best_cwnd, indigo_cwnd')

    address = ('0.0.0.0', int(sys.argv[1]))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(address)
    s.listen(5)

    sys.stderr.write('expert server is listenning')

    # thread.start_new_thread(thread_fun, (1,))
    cal_thread = threading.Thread(target=thread_fun)
    cal_thread.start()

    ss, addr = s.accept()

    # start new process to show performance
    perf_process.start()

    running = 1
    while True:
        if cal_thread.isAlive() is not True:
            sys.stderr.write('start new thread')
            cal_thread = threading.Thread(target=thread_fun)
            cal_thread.start()
        indigo_cwnd = float(ss.recv(16))
        exp.flag = 1

    perf_process.terminate()
    ss.close()
    s.close()
