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
from subprocess import Popen
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.topo import Topo


class CreateEmulatorTopo(Topo):
    ''' Topology:
    h1 (tg_sender) ------------+                            +--- h2 (tg_receiver)
                               |---- s1 ==== s2 ---- s3 ----|
    host (sender) --- veth1 ---+     bottleneck             +--- veth2 --- h3 (receiver)
    '''
    def build(self, args):
        switch1 = self.addSwitch('s1')
        switch2 = self.addSwitch('s2')
        switch3 = self.addSwitch('s3')

        host1 = self.addHost('h1', ip='192.168.42.1/24')
        host2 = self.addHost('h2', ip='192.168.42.2/24')
        host3 = self.addHost('h3', ip='192.168.42.222/24')

        # set delay and max_queue_size together
        self.addLink(switch1, switch2,
                     bw=args.bw, delay=args.delay, loss=args.loss,
                     max_queue_size=args.max_queue_size)  # s1-eth1
        self.addLink(switch2, switch3)

        self.addLink(host1, switch1, bw=1000)
        self.addLink(host2, switch3, bw=1000)  # s3-eth2
        self.addLink(host3, switch3, bw=1000)  # s3-eth3


def start_emulator(args):
    topo = CreateEmulatorTopo(args)
    net = Mininet(topo, link=TCLink)
    net.start()
    net.pingAll()

    Popen('ip link add veth1 type veth peer name veth2', shell=True)
    Popen('ifconfig veth1 up', shell=True)
    Popen('ifconfig veth2 up', shell=True)
    Popen('ifconfig veth1 192.168.42.111', shell=True)  # sender's IP
    Popen('ovs-vsctl add-port s1 veth2', shell=True)

    # run mininet CLI
    CLI(net)

    # stop and clean
    Popen('ip link delete veth1', shell=True)  # don't need to delete veth2
    net.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bw', type=float, help='a float in Mbit/s')
    parser.add_argument('delay', help='a string with units (e.g. 5ms)')
    parser.add_argument('loss', type=int,
                        help='a percentage (integer between 0 and 100)')
    parser.add_argument('max_queue_size', type=int, help='packets (integer)')
    args = parser.parse_args()

    # tell mininet to print useful information
    setLogLevel('info')

    start_emulator(args)


if __name__ == '__main__':
    main()
