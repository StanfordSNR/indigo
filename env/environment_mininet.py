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
import os
import sys
import signal
import subprocess
import project_root
import time
from os import path
from helpers.helpers import get_open_udp_port, get_open_tcp_port, check_pid
from sender import Sender


def get_mininet_env_param(train):
    total_tp_set = []
    total_env_name_set = []

    if train:
        option_name = 'train_env'
    else:
        option_name = 'test_env'

    cfg = ConfigParser.ConfigParser()
    cfg_path = os.path.join(project_root.DIR, 'config.ini')
    cfg.read(cfg_path)

    env = cfg.options(option_name)
    for opt in env:
        env_param, tp_set_param = ast.literal_eval(cfg.get(option_name, opt))
        total_tp_set.append(ast.literal_eval(cfg.get('global', tp_set_param)))
        total_env_name_set.append(env_param)

    return total_tp_set, total_env_name_set


class Environment_Mininet(object):
    def __init__(self, traffic_shape_set, env_set, train):

        self.traffic_shape = 1  # default traffic_shape
        self.traffic_shape_set = traffic_shape_set
        self.traffic_shape_set_len = len(self.traffic_shape_set)
        self.traffic_shape_set_index_1 = 0
        self.traffic_shape_set_index_2 = 0

        self.env_set = env_set
        self.env_set_len = len(self.env_set)
        self.env_set_index = 0

        self.state_dim = Sender.state_dim
        self.action_cnt = Sender.action_cnt

        self.train = train

        self.done = False

        # variables below will be filled in during setup
        self.sender = None
        self.receiver = None
        self.emulator = None
        self.expert = None
        self.expert_server = None

    def set_expert(self, expert):
        self.expert = expert

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before calling reset()."""

        self.sample_action = sample_action

    def all_tasks_done(self):
        ret = self.done
        if ret:
            self.done = False
        return ret

    def reset(self):
        """Must be called before running rollout()."""

        self.cleanup()

        self.port = get_open_udp_port()

        if self.train:
            sys.stderr.write('start emulator expert server\n')
            expert_server_path = path.join(project_root.DIR, 'dagger', 'expert_server.py')
            for i in xrange(5):
                self.tcp_port = get_open_tcp_port()
                cmd = ('python ' + expert_server_path + ' {} '.format(self.tcp_port)).split(' ')
                self.expert_server = subprocess.Popen(cmd, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w')) #
                if check_pid(self.expert_server.pid):
                    sys.stderr.write('start expert server successfully\n')
                    break
                else:
                    sys.stderr.write('start expert server failed, try again...\n')

        else:
            sys.stderr.write('start emulator perf server\n')
            expert_server_path = path.join(project_root.DIR, 'dagger', 'perf_server.py')
            for i in xrange(5):
                self.tcp_port = get_open_tcp_port()
                cmd = ('python ' + expert_server_path + ' {} {} {} {}'.format(self.tcp_port, self.env_set_index, self.traffic_shape_set_index_1, self.traffic_shape_set_index_2)).split(' ')
                self.expert_server = subprocess.Popen(cmd, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
                if check_pid(self.expert_server.pid):
                    sys.stderr.write('start perf server successfully\n')
                    break
                else:
                    sys.stderr.write('start perf server failed, try again...\n')

        # start sender as an instance of Sender class
        self.sender = Sender(self.port, train=self.train)
        self.sender.set_sample_action(self.sample_action)

        total_tp_set, total_env_name_set = get_mininet_env_param(self.train)
        curr_env_name = total_env_name_set[self.env_set_index]
        curr_tp_name = total_tp_set[self.traffic_shape_set_index_1][self.traffic_shape_set_index_2]
        log_file_postfix = '{}_tp_{}.log'.format(curr_env_name, curr_tp_name)

        if not self.train:
            self.sender.set_test_name(log_file_postfix)

        # start emulator-mininet
        sys.stderr.write('start mininet emulator\n')
        cmd_para = ' ' + self.env_set[self.env_set_index][0] + ' ' + self.env_set[self.env_set_index][1] + \
                   ' ' + self.env_set[self.env_set_index][2] + ' ' + self.env_set[self.env_set_index][3]
        sys.stderr.write(cmd_para+'\n')
        emulator_path = path.join(project_root.DIR, 'netEmulator', 'emulator_topo.py')
        cmd = ['python', emulator_path, self.env_set[self.env_set_index][0], self.env_set[self.env_set_index]
               [1], self.env_set[self.env_set_index][2], self.env_set[self.env_set_index][3]]
        self.emulator = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print self.emulator.stdout.read()
        # time.sleep(1.5)

        self.traffic_shape = self.traffic_shape_set[self.traffic_shape_set_index_1][self.traffic_shape_set_index_2]

        if self.traffic_shape == -1:
            sys.stderr.write('start iperf server\n')
            self.emulator.stdin.write('h2 iperf3 -s &\n')
            self.emulator.stdin.flush()
            # time.sleep(0.5)

            self.emulator.stdin.write('h1 iperf3 -c 192.168.42.2 -C bbr -t 50 -M 2902&\n')
            self.emulator.stdin.flush()
        else:
            sys.stderr.write('start traffic generator\n')
            # tg-receiver:
            tg_receiver_path = path.join(project_root.DIR, 'trafficGenerator', 'receiver.py')
            self.emulator.stdin.write('h2 python ' + tg_receiver_path + ' 192.168.42.2 6666 &\n')
            self.emulator.stdin.flush()
            # time.sleep(0.5)

            # start traffic generator (tg)
            # tg-sender: PARAMETER ip port NIC traffic_shape duration
            sys.stderr.write('Traffic shape index is {} \n'.format(self.traffic_shape))
            tg_sender_path = path.join(project_root.DIR, 'trafficGenerator', 'sender.py')
            self.emulator.stdin.write('h1 python ' + tg_sender_path + ' 192.168.42.2 6666 h1-eth0 {} 0 &\n'.format(self.traffic_shape))
            self.emulator.stdin.flush()

        if self.traffic_shape_set_index_2 == (len(self.traffic_shape_set[self.traffic_shape_set_index_1]) - 1):
            self.env_set_index = self.env_set_index + 1
            self.traffic_shape_set_index_1 = self.traffic_shape_set_index_1 + 1
            self.traffic_shape_set_index_2 = 0
            if self.env_set_index == self.env_set_len:
                self.done = True
                self.env_set_index = 0
                self.traffic_shape_set_index_1 = 0
                self.traffic_shape_set_index_2 = 0
        else:
            self.traffic_shape_set_index_2 = self.traffic_shape_set_index_2 + 1

        # start receiver
        sys.stderr.write('Start receiver\n')
        receiver_path = path.join(project_root.DIR, 'env', 'run_receiver.py')
        if self.train:
            self.emulator.stdin.write('h3 python ' + receiver_path + ' 192.168.42.111 ' + str(self.port) + ' & \n')
        else:
            self.emulator.stdin.write('h3 python ' + receiver_path + ' 192.168.42.111 ' + str(self.port) + ' -t ' + log_file_postfix + ' & \n')
        self.emulator.stdin.flush()

        if self.train:
            time.sleep(1)
            for i in xrange(10):
                ret = self.expert.connect_expert_server(self.tcp_port)
                if ret == 0:
                    break
                time.sleep(0.5)
        else:
            time.sleep(1)
            for i in xrange(5):
                ret = self.expert.connect_perf_server(self.tcp_port)
                if ret == 0:
                    break
                time.sleep(0.5)
            self.sender.set_perf_client(self.expert)

        # sender completes the handshake sent from receiver
        sys.stderr.write('Starting sender...\n')
        self.sender.handshake()

        sys.stderr.write('env reset done\n')

    def rollout(self):
        """Run sender in env, get final reward of an episode, reset sender."""

        sys.stderr.write('Obtaining an episode from environment...\n')
        ret = self.sender.run()
        return ret

    def cleanup(self):

        if self.emulator:
            if not self.train:
                self.emulator.stdin.write('h3 pkill -f run_receiver\n')
                time.sleep(0.5)
            self.emulator.stdin.write('quit()\n')
            self.emulator = None
            time.sleep(3)  # wait for the mininet to be closed completely

        if self.expert_server:
            if self.train:
                subprocess.Popen('pkill -f expert_server', shell=True)
            else:
                subprocess.Popen('pkill -f perf_server', shell=True)
            self.expert_server = None

        if self.expert:
            self.expert.cleanup()

        if self.sender:
            self.sender.cleanup()
            self.sender = None

        if self.receiver:
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
            finally:
                self.receiver = None

    def get_best_cwnd(self):
        best_rate, best_cwnd = self.expert.get_network_state()
        return best_cwnd
