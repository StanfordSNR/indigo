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


import subprocess
import time
import datetime
import thread


#######################
#This file is not used#
#######################

class Device_QDISC():
    def __init__(self, dev_name):

        self.dev_name = dev_name

        self.sent_bytes  = 0
        self.sent_pkt    = 0
        self.dropped_pkt = 0
        self.record_time = datetime.datetime.now()

        #Emulator configuration
        self.BW_CAPACITY    = 1000 #Mbps
        self.MAX_QUEUE_SIZE = 1000 #number
        self.LOSS_RATE      = 0 #%
        self.MIN_RTT        = 10 #ms

    def get_emulator_configuration(self):

        try:
            out_bytes = subprocess.check_output(['tc','-p', '-s', '-d', 'qdisc', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            code      = e.returncode
            raise e
        out_text    = out_bytes.decode('utf-8')
        out_slice   = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('qdisc netem')!=-1])

        if target_line.find('limit')!=-1:
            self.MAX_QUEUE_SIZE = int(target_line.split(' ')[6])
        if target_line.find('delay')!=-1:
            self.MIN_RTT = 2*float(target_line.split(' ')[8][:-2])
        if target_line.find('loss')!=-1:
            self.LOSS_RATE = float(target_line.split(' ')[10][:-1])

        try:
            out_bytes = subprocess.check_output(['tc','-p', '-s', '-d', 'class', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            code      = e.returncode
            raise e
        out_text    = out_bytes.decode('utf-8')
        out_slice   = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('class htb')!=-1])

        self.BW_CAPACITY = int(target_line.split(' ')[11][:-4])

        print 'Emulator Configuration: max queue size(pkt), loss rate(%), min RTT(ms), bandwidth capacity(Mbps)'
        print self.MAX_QUEUE_SIZE, self.LOSS_RATE, self.MIN_RTT, self.BW_CAPACITY

    def get_network_state(self):
        best_rate = 0
        best_cwnd=0
        _record_time = datetime.datetime.now()

        try:
            out_bytes = subprocess.check_output(['tc','-p', '-s', '-d', 'qdisc', 'show', 'dev', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output       # Output generated before error
            code      = e.returncode   # Return code
            raise e


        out_text = out_bytes.decode('utf-8')
        out_slice = out_text.split('\n')

        _sent_bytes  = int(out_slice[1].split(' ')[2])
        _sent_pkt    = int(out_slice[1].split(' ')[4])
        _dropped_pkt = int(out_slice[1].split(' ')[7][:-1])
        #_backlog_b  = int(out_slice[2].split(' ')[2][:-1])
        _backlog_p   = int(out_slice[2].split(' ')[3][:-1])
        #print _sent_bytes,_sent_pkt,_dropped_pkt,_backlog_p

        if self.sent_bytes != 0:
            record_time = (_record_time - self.record_time).microseconds

            available_bw        = self.BW_CAPACITY - 8*(_sent_bytes - self.sent_bytes)/(1.0*record_time)
            queueing_factor     = 1.0 * _backlog_p / self.MAX_QUEUE_SIZE
            non_congestion_loss = (_sent_pkt - self.sent_pkt) * 1.0 * self.LOSS_RATE / 100
            congestion_loss     = (_dropped_pkt-self.dropped_pkt)-non_congestion_loss
            #print time, available_bw, queueing_factor, non_congestion_loss, congestion_loss
            #best_rate, best_cwnd = self.calculate_oracle(available_bw, queueing_factor, non_congestion_loss, congestion_loss)

        else :
            available_bw = 0
            queueing_factor = 0
            non_congestion_loss = 0
            congestion_loss = 0

        self.sent_bytes  = _sent_bytes
        self.sent_pkt    = _sent_pkt
        self.dropped_pkt = _dropped_pkt
        self.record_time = _record_time

        return available_bw, queueing_factor, non_congestion_loss, congestion_loss

class Device_IFCONFIG():
    def __init__(self, dev_name):
        self.dev_name = dev_name

        self.rx_packets = 0
        self.record_time = datetime.datetime.now()

    def get_rx_rate(self):
        try:
            out_bytes = subprocess.check_output(['ifconfig', self.dev_name])
        except subprocess.CalledProcessError as e:
            out_bytes = e.output
            code      = e.returncode
            raise e
        out_text    = out_bytes.decode('utf-8')
        out_slice   = out_text.split('\n')
        target_line = ''.join([x for x in out_slice if x.find('RX packets')!=-1])


        _rx_packets = int(target_line.split(':')[1].split(' ')[0])
        _record_time = datetime.datetime.now()
        if self.rx_packets != 0:
            rx_rate = 1.0*(_rx_packets - self.rx_packets) * (1380+8+20+18)*8 / (_record_time - self.record_time).microseconds
        else:
            rx_rate = 0

        self.rx_packets = _rx_packets
        self.record_time = _record_time
        return rx_rate


class Expert_Mininet():
    def __init__(self, dev_bottleneck, dev_traffic_generator):

        self.dev_bottleneck = Device_QDISC(dev_bottleneck)
        self.dev_traffic_generator = Device_IFCONFIG(dev_traffic_generator)

    def init_configure(self):
        try:
            time.sleep(2) #wait mininet to start up
            self.dev_bottleneck.get_emulator_configuration()
        except:
            print 'get emulator coniguration error Mininet Emulator does not start'
            return

    def calculate_oracle(self):
        '''
        TODO:
        Method 1: calculate the best sending rate and best_cwnd
        Method 2: give a score as the reward"
        '''
        try:
            available_bw_bn, queueing_factor_bn, non_congestion_loss_bn, congestion_loss_bn = self.dev_bottleneck.get_network_state()
        except:
            print 'Mininet Emulator does not start'
            return
        try:
            tg_traffic = self.dev_traffic_generator.get_rx_rate()
        except:
            print 'ifconfig error'
            return

        #just for test
        if available_bw_bn < 0:
            available_bw_bn = 0
        #print tg_traffic
        #print self.dev_bottleneck.MIN_RTT
        best_send_rate =  self.dev_bottleneck.BW_CAPACITY - (tg_traffic)#Mbps
        best_cwnd      = 1000 * best_send_rate * self.dev_bottleneck.MIN_RTT #Mbps * ms = 10^6 b/s * 10^(-3) s = b
        best_cwnd      = best_cwnd / (8*(1400+28+8+20+18)) #packet number, each packet is 1400B in sender
        #print best_send_rate
        #print '----',best_cwnd
        return best_cwnd


if __name__ == '__main__':

    exp = Expert_Mininet('s1-eth1','s1-eth2')
    exp.init_configure()
    #exp.get_emulator_configuration()
    while True:
        exp.calculate_oracle()
        time.sleep(0.048) # 0.008 is a experienced value

