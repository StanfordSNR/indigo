# Copyright 2018 Francis Y. Yan, Jestin Ma
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


import datetime
import socket
import subprocess
import sys
import threading

import context
from helpers.utils import min_x_max
from message import Message
from oracle import AggressiveBDP, MoreAggressiveBDP
from policy import Policy


class DeviceQdisc():
    """ get dev state via TC qdisc """

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

    def get_configuration(self):
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

        # transfer mininet queue size to switch queue size since origin includes the delay of netem
        self.MAX_QUEUE_SIZE = self.MAX_QUEUE_SIZE - self.BW_CAPACITY*(self.MIN_RTT/2)*1000/Message.total_size/8
        print 'Emulator Configuration: max queue size(pkt), loss rate(%), min RTT(ms), bandwidth capacity(Mbps)'
        print self.MAX_QUEUE_SIZE, self.LOSS_RATE, self.MIN_RTT, self.BW_CAPACITY

    def get_curr_state(self, flag):
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
            # _sent_pkt = int(out_slice[1].split(' ')[4])
            _dropped_pkt = int(out_slice[1].split(' ')[7][:-1])
            # _backlog_b  = int(out_slice[2].split(' ')[2][:-1])
            _backlog_pkts = int(out_slice[2].split(' ')[3][:-1])
            # transfer backlog pkts to 'real' queueing pkts since origin includes the delay of netem
            _backlog_pkts_queueing = max(0, _backlog_pkts-self.BW_CAPACITY*(self.MIN_RTT/2)*1000/Message.total_size/8)
            # print _sent_bytes,_sent_pkt,_dropped_pkt,_backlog_p
        except IndexError:
            return 0, 0, 0, 0, 0

        if self.sent_bytes != 0:
            delta_time = (_record_time - self.record_time).microseconds
            if not delta_time:
                return 0, 0, 0, 0, 0

            available_bw = self.BW_CAPACITY - 8.0*(_sent_bytes - self.sent_bytes)/delta_time
            queueing_factor = 1.0 * _backlog_pkts_queueing / self.MAX_QUEUE_SIZE
            congestion_loss_pkts = int((1.0*(_sent_bytes-self.sent_bytes)/Message.total_size+(_dropped_pkt-self.dropped_pkt))
                                       - 100.0*(_sent_bytes-self.sent_bytes)/Message.total_size/(100.0-self.LOSS_RATE))
            random_loss_pkts = _dropped_pkt - self.dropped_pkt - congestion_loss_pkts
            # print time, available_bw, queueing_factor, random_loss_pkts, congestion_loss_pkts
        else:
            available_bw = 0
            queueing_factor = 0
            random_loss_pkts = 0
            congestion_loss_pkts = 0

        if flag == 1:
            self.sent_bytes = _sent_bytes
            # self.sent_pkt = _sent_pkt
            self.dropped_pkt = _dropped_pkt
            self.record_time = _record_time

        return available_bw, queueing_factor, random_loss_pkts, congestion_loss_pkts, _backlog_pkts


class DeviceIfconfig():
    """ get dev state via ifconfig """

    def __init__(self, dev_name):
        self.dev_name = dev_name

        self.rx_packets = 0
        self.tx_packets = 0

        self.rx_record_time = datetime.datetime.now()
        self.tx_record_time = datetime.datetime.now()

    def get_rx_rate(self, flag):
        _record_time = datetime.datetime.now()
        try:
            out_bytes = subprocess.check_output(['ifconfig', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e

        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('RX packets') != -1])
        _rx_packets = int(target_line.split(':')[1].split(' ')[0])

        if self.rx_packets != 0:
            delta_time = (_record_time-self.rx_record_time).microseconds
            if not delta_time:
                return 0

            rx_rate = 8.0*(_rx_packets-self.rx_packets)*Message.total_size / delta_time
        else:
            rx_rate = 0
            self.rx_packets = _rx_packets

        if flag == 1:
            self.rx_packets = _rx_packets
            self.rx_record_time = _record_time

        return rx_rate

    def get_tx_rate(self, flag):
        _record_time = datetime.datetime.now()
        try:
            out_bytes = subprocess.check_output(['ifconfig', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            raise e

        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('TX packets') != -1])
        _tx_packets = int(target_line.split(':')[1].split(' ')[0])

        if self.tx_packets != 0:
            delta_time = (_record_time-self.tx_record_time).microseconds
            if not delta_time:
                return 0

            tx_rate = 8.0*(_tx_packets-self.tx_packets)*Message.total_size / delta_time
        else:
            tx_rate = 0
            self.tx_packets = _tx_packets

        if flag == 1:
            self.tx_packets = _tx_packets
            self.tx_record_time = _record_time

        return tx_rate

class ExpertServer():
    """ server for expert, call oracle to get best cwnd """

    def __init__(self, dev_bottleneck, dev_traffic_generator):
        self.dev_bottleneck = DeviceQdisc(dev_bottleneck)
        self.dev_traffic_generator = DeviceIfconfig(dev_traffic_generator)
        self.flag = 1
        self.oracle = None

    def init_configure(self):
        try:
            # time.sleep(2)  # wait mininet to start up
            self.dev_bottleneck.get_configuration()
        except subprocess.CalledProcessError:
            sys.stderr.write('get emulator coniguration error, mininet does not start')
            return
        self.net_config = self.dev_bottleneck.BW_CAPACITY, self.dev_bottleneck.MIN_RTT, self.dev_bottleneck.MAX_QUEUE_SIZE
        # set policy of oracle
        self.set_oracle(MoreAggressiveBDP(self.net_config))

    def set_oracle(self, oralce):
        self.oracle = oralce

    def get_oracle(self):
        flag = self.flag
        if self.flag == 1:
            self.flag = 0
        try:
            curr_state = self.dev_bottleneck.get_curr_state(flag)
            available_bw_bn, queueing_factor_bn, random_loss_bn, congestion_loss_bn, queue_size_bn = curr_state
        except subprocess.CalledProcessError:
            sys.stderr.write('mininet does not start')
            return
        try:
            throughput_tg = self.dev_traffic_generator.get_tx_rate(flag)
        except subprocess.CalledProcessError:
            sys.stderr.write('ifconfig error')
            return

        available_bw_bn = min_x_max(0, available_bw_bn, self.dev_bottleneck.BW_CAPACITY)
        throughput_tg = min_x_max(0, throughput_tg, self.dev_bottleneck.BW_CAPACITY)
        queueing_factor_bn = min_x_max(0.0, queueing_factor_bn, 1.0)
        random_loss_bn = min_x_max(0, random_loss_bn, self.dev_bottleneck.MAX_QUEUE_SIZE)

        net_info = available_bw_bn, throughput_tg, queueing_factor_bn, random_loss_bn, congestion_loss_bn, queue_size_bn
        best_cwnd = self.oracle.get_oracle(net_info)

        return max(Policy.min_cwnd, best_cwnd)


def thread_fun():
    global expert, cwnd, running
    while (running == 0):
        pass
    expert.init_configure()
    while True:
        cwnd = expert.get_oracle()


if __name__ == '__main__':
    address = ('0.0.0.0', int(sys.argv[1]))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(address)
    s.listen(5)

    print 'expert server is listenning'

    #expert = ExpertServer('s1-eth1', 's1-eth2')
    expert = ExpertServer('s1-eth1', 's3-eth2')
    cwnd = 0
    running = 0

    cal_thread = threading.Thread(target=thread_fun)
    cal_thread.start()

    ss, addr = s.accept()
    running = 1

    while True:
        if cal_thread.isAlive() is not True:
            print 'start new thread'
            cal_thread = threading.Thread(target=thread_fun)
            cal_thread.start()
        ra = ss.recv(1024)
        if ra == 'Current best cwnd?':
            ss.send(str(cwnd))
            expert.flag = 1

    ss.close()
    s.close()
