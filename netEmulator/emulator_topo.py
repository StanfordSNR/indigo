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


from mininet.topo import Topo
from mininet.net import Mininet
from mininet.util import dumpNodeConnections
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.link import TCLink
import os
import sys

class CreateEmulatorTopo(Topo):
    '''
    Topo:
                   h1(traffic generator sender) --- s1 --- s2 --- s3 --- h2(traffic generator receiver)"
       sender ---veth-pair|             |---- h3(recever)
    '''
    def build(self, Bw = 1000, Delay= '5ms', Loss=0, Queue_size=1500):
        switch1 = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')
        switch3 = self.addSwitch('s3')
        #delay and max queue size should be configured together, based on the min rtt and max rtt in the real experiments. Bw should also from real network
        self.addLink(switch1,switch2, bw=float(Bw), delay=Delay,loss = int(Loss), max_queue_size=int(Queue_size)) #s1-eth1
        self.addLink(switch2,switch3)

        host1 = self.addHost('h1',ip='192.168.42.1/24')
        host2 = self.addHost('h2',ip='192.168.42.2/24')
        host3 = self.addHost('h3',ip='192.168.42.222/24')
        self.addLink(host1, switch1,bw=1000)
        self.addLink(host2, switch3,bw=1000) #s3-eth2
        self.addLink(host3, switch3,bw=1000) #s3-eth3


def StartEmulator(_bw, _delay, _loss, _queue_size):
    "Create network topology"
    topo = CreateEmulatorTopo(Bw=_bw, Delay=_delay, Loss=_loss, Queue_size=_queue_size)
    net = Mininet(topo,link=TCLink)
    net.start()
    print "Testing network connectivity"
    net.pingAll()

    os.popen('ip link add veth1 type veth peer name veth2')
    os.popen('ifconfig veth1 up')
    os.popen('ifconfig veth2 up')
    os.popen('ifconfig veth1 192.168.42.111') #sender's port
    os.popen('ovs-vsctl add-port s1 veth2')

    CLI(net)
    net.stop()
    os.popen('ip link delete veth1') #Do not need to delete veth2

if __name__ == '__main__':
    # Tell mininet to print useful information
    setLogLevel('info')
    # bw, delay, loss, queue
    StartEmulator(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
