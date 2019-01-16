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

import argparse
import time
import sys
from os import path
from subprocess import call

import context
from dagger.message import Message
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.topo import Topo


class CreateEmulatorTopo(Topo):
    ''' Topology:
    h1 (tg_sender) ------------+                            +--- h2 (tg_receiver)
                               |---- s1 ==== s2 ---- s3 ----|
    host (sender) --- veth1 ---+     bottleneck             +--- h3 (receiver)
    '''
    def build(self, _bw, _delay, _loss, _max_queue_size):
        switch1 = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')
        switch3 = self.addSwitch('s3')

        host1 = self.addHost('h1', ip='192.168.42.1/24')
        host2 = self.addHost('h2', ip='192.168.42.2/24')
        host3 = self.addHost('h3', ip='192.168.42.222/24')

        # set delay and max_queue_size together
        self.addLink(switch1, switch2,
                     bw=_bw, delay=_delay, loss=_loss,
                     max_queue_size=_max_queue_size)  # s1-eth1
        self.addLink(switch2, switch3)

        self.addLink(host1, switch1, bw=1000)
        self.addLink(host2, switch3, bw=1000)  # s3-eth2
        self.addLink(host3, switch3, bw=1000)  # s3-eth3


class Emulator(object):
    def __init__(self):
        self.network = None
        self.tg_sender_host = None
        self.tg_receiver_host = None
        self.receiver_host = None

        self.try_time = 3

    def start_network(self, bw, delay, loss, qsize):
        try:
            # sys.stderr.write('\nStep #1: start mininet emulator: {} {} {} {}\n'
            #                  .format(bw, delay, loss, qsize))

            topo = CreateEmulatorTopo(float(bw), str(delay), int(loss), int(qsize))
            self.network = Mininet(topo, link=TCLink)
            self.network.start()

            # add veth1 and veth2 for sender
            call('ip link add veth1 type veth peer name veth2', shell=True)
            call('ifconfig veth1 up', shell=True)
            call('ifconfig veth2 up', shell=True)
            call('ifconfig veth1 192.168.42.111', shell=True)  # sender's IP
            call('ovs-vsctl add-port s1 veth2', shell=True)

            self.tg_sender_host = self.network.getNodeByName('h1')
            self.tg_receiver_host = self.network.getNodeByName('h2')
            self.receiver_host = self.network.getNodeByName('h3')
        except Exception:
            sys.stderr.write('Start mininet error. Retrying...')
            call('ip link delete veth1', shell=True)
            call('mn -c >/dev/null 2>&1', shell=True)
            if self.try_time > 0:
                time.sleep(10)
                self.try_time = self.try_time - 1
                return self.start_network(bw, delay, loss, qsize)
            else:
                return -1
        return 0

    def exe_cmd(self, host, cmd):
        if host:
            host.sendCmd(cmd)
            host.waitOutput()

    def start_receiver(self, port):
        # sys.stderr.write('Step #3: recervier\n')
        receiver_path = path.join(context.base_dir, 'dagger', 'receiver.py')
        cmd = 'python ' + receiver_path + ' ' + str(port) + ' &'
        self.exe_cmd(self.receiver_host, cmd)

    def start_tg(self, generator):
        if type(generator) is str and generator.startswith('iperf'):
            # use BBR/Cubic/... as the background traffic with iperf/iperf3

            # sys.stderr.write('Step #2: start {} as traffic generator\n'.format(generator))
            tool, method = generator.split('.')
            param = 'Z' if tool == 'iperf' else 'C'

            # set tso off
            self.exe_cmd(self.tg_sender_host, 'ethtool -K h1-eth0 tso off')

            self.exe_cmd(self.tg_receiver_host, '{} -s &'.format(tool))
            time.sleep(0.1)
            self.exe_cmd(self.tg_sender_host, '{} -c 192.168.42.2 -{} {} -M {} -t 2000 &'
                         .format(tool, param, method, Message.total_size))
        else:
            # sys.stderr.write('Step #2: start traffic generator: ')
            # tg-receiver:
            tg_receiver_path = path.join(
                context.base_dir, 'traffic-generator', 'receiver.py')
            self.exe_cmd(self.tg_receiver_host, 'python ' + tg_receiver_path + ' 192.168.42.2 6666 &')
            time.sleep(0.1)

            # tg-sender: parameters: ip port NIC -s [sketch] -c cycle -l lifetime
            sketch, cycle = generator
            # sys.stderr.write('{}\n'.format(sketch))
            # one_way_delay = convert_to_seconds(self.env_set[self.env_set_idx][1])
            # lifetime = Policy.steps_per_episode * one_way_delay * 2 / Policy.action_frequency * 1.5
            lifetime = 1000  # 0 stands for run forever
            tg_sender_path = path.join(
                context.base_dir, 'traffic-generator', 'sender.py')
            self.exe_cmd(self.tg_sender_host, 'python {} 192.168.42.2 6666 h1-eth0 -s "{}" -c {} -l {} &'
                                              .format(tg_sender_path, sketch, cycle, lifetime))
            time.sleep(0.1)

    def stop_all(self):
        self.exe_cmd(self.tg_sender_host, 'pkill -f sender')
        self.exe_cmd(self.tg_sender_host, 'pkill iperf3')
        self.exe_cmd(self.tg_receiver_host, 'pkill -f receiver')
        self.exe_cmd(self.tg_receiver_host, 'pkill iperf3')
        self.exe_cmd(self.receiver_host, 'pkill -f receiver')

        call('ip link delete veth1', shell=True)  # don't need to delete veth2

        if self.network is not None:
            self.network.stop()
        else:
            call('ip link delete veth1', shell=True)
            call('mn -c >/dev/null 2>&1', shell=True)

        self.network = None
        self.tg_sender_host = None
        self.tg_receiver_host = None
        self.receiver_host = None


def start_emulator(bw, delay, loss, qsize):
    topo = CreateEmulatorTopo(bw, delay, loss, qsize)
    net = Mininet(topo, link=TCLink)
    net.start()
    net.pingAll()

    call('ip link add veth1 type veth peer name veth2', shell=True)
    call('ifconfig veth1 up', shell=True)
    call('ifconfig veth2 up', shell=True)
    call('ifconfig veth1 192.168.42.111', shell=True)  # sender's IP
    call('ovs-vsctl add-port s1 veth2', shell=True)

    # set tso off
    net.getNodeByName('h1').sendCmd('ethtool -K h1-eth0 tso off')

    # run mininet CLI
    CLI(net)

    # stop and clean
    call('ip link delete veth1', shell=True)  # don't need to delete veth2
    net.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bw', type=float, help='a float in Mbit/s')
    parser.add_argument('delay', help='a string with units (e.g. 5ms)')
    parser.add_argument('loss', type=int, help='a percentage (integer between 0 and 100)')
    parser.add_argument('max_queue_size', type=int, help='packets (integer)')
    args = parser.parse_args()

    # tell mininet to print useful information
    setLogLevel('info')

    start_emulator(args.bw, args.delay, args.loss, args.max_queue_size)


if __name__ == '__main__':
    main()
