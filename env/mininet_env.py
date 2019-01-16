#!/usr/bin/env python

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

import time
from os import path
from subprocess import Popen

import context
from dagger.policy import Policy
from dagger.sender import Sender
from helpers.utils import DEVNULL, Config, check_pid, get_open_port
from mininet_topo import Emulator


class MininetEnv(object):
    def __init__(self, env_set, tpg_set, train):
        # traffic pattern group is defined in config.ini and parsed by helper.utils
        self.tpg_set = tpg_set
        self.tpg_set_len = len(self.tpg_set)
        self.tpg_set_idx_env = 0
        self.tpg_set_idx_gen = 0

        self.env_set = env_set
        self.env_set_len = len(self.env_set)
        self.env_set_idx = 0

        self.state_dim = Policy.state_dim
        self.action_cnt = Policy.action_cnt

        self.train = train

        self.done = False

        # variables below will be filled in during setup
        self.sender = None
        self.policy = None
        self.receiver = None
        self.emulator = None
        self.expert_client = None
        self.expert_server = None
        self.perf_client = None
        self.perf_server = None

    def set_expert_client(self, expert_client):
        self.expert_client = expert_client

    def set_perf_client(self, perf_client):
        self.perf_client = perf_client

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before calling reset()."""

        self.sample_action = sample_action

    def is_all_tasks_done(self):
        ret = self.done
        if ret:
            self.done = False
        return ret

    def __update_env_tpg(self):
        if self.tpg_set_idx_gen == len(self.tpg_set[self.tpg_set_idx_env]) - 1:
            # this env is done
            self.env_set_idx = self.env_set_idx + 1
            self.tpg_set_idx_env = self.tpg_set_idx_env + 1
            self.tpg_set_idx_gen = 0
            # all env and traffic shape are done
            if self.env_set_idx == self.env_set_len:
                self.done = True
                self.env_set_idx = 0
                self.tpg_set_idx_env = 0
                self.tpg_set_idx_gen = 0
        else:
            # get next traffic shape for this env
            self.tpg_set_idx_gen = self.tpg_set_idx_gen + 1

    def __start_expert(self):
        expert_server_path = path.join(
                context.base_dir, 'dagger', 'expert_server.py')
        # try 3 times at most to ensure expert server is started normally
        expert_server_flag = False
        for i in xrange(3):
            self.expert_server_port = get_open_port()
            cmd = ['python', expert_server_path, str(self.expert_server_port)]
            self.expert_server = Popen(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
            if check_pid(self.expert_server.pid):
                expert_server_flag = True
                # sys.stderr.write(
                #     'Step #4: start expert server (PID: {}), '.format(
                #         self.expert_server.pid))
                break
        if not expert_server_flag:  # start expert server failed
            return -1
        # wait for server start completely
        time.sleep(1)

        if self.expert_client is None:
            # sys.stderr.write('\nNo expert_client set\n')
            return -1

        ret = -1
        for i in xrange(3):
            ret = self.expert_client.connect_expert_server(self.expert_server_port)
            if ret == 0:
                # sys.stderr.write('connect to expert server successfully\n')
                break
            time.sleep(0.5)
        if ret == -1:
            # sys.stderr.write('connect to expert server failed\n')
            return -1

        return 0

    def __start_perf(self):
        perf_server_path = path.join(context.base_dir, 'dagger', 'perf_server.py')
        # try 3 times at most to ensure perf server is started normally
        perf_server_flag = False
        for i in xrange(3):
            self.perf_server_port = get_open_port()
            cmd = ['python', perf_server_path, str(self.perf_server_port),
                   self.env_set_idx, self.tpg_set_idx_env, self.tpg_set_idx_gen]
            self.perf_server = Popen(cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, close_fds=True)
            if check_pid(self.perf_server.pid):
                perf_server_flag = True
                # sys.stderr.write('Step #4: start perf server successfully\n')
                break
        if not perf_server_flag:  # start perf server failed
            return -1
        # wait for server start completely
        time.sleep(1)

        if self.perf_client is None:
            # sys.stderr.write('\nNo perf_client set\n')
            return -1

        ret = -1
        for i in xrange(3):
            ret = self.perf_client.connect_perf_server(self.perf_server_port)
            if ret == 0:
                # sys.stderr.write('connect to perf server successfully\n')
                break
            time.sleep(0.5)
        if ret == -1:
            # sys.stderr.write('connect to perf server failed\n')
            return -1

        return 0

    def reset(self):
        """Must be called before running rollout()."""

        self.port = get_open_port()

        # STEP 1: start mininet emulator
        self.emulator = Emulator()
        ret = self.emulator.start_network(*self.env_set[self.env_set_idx])
        if ret == -1:  # start emulator failed
            self.__update_env_tpg()
            return -1
        time.sleep(0.1)

        # STEP 2: start traffic generator
        generator = self.tpg_set[self.tpg_set_idx_env][self.tpg_set_idx_gen]
        self.emulator.start_tg(generator)

        # STEP 3: start receiver
        self.port = get_open_port()
        self.emulator.start_receiver(self.port)

        # STEP 4: start expert or perf in train or test mode
        ret = 0
        if self.train:
            ret = self.__start_expert()
        elif Config.perf:
            ret = self.__start_perf()

        if ret == -1:  # start expert or perf failed
            self.__update_env_tpg()
            return -1

        # STEP 5: start sender
        # sys.stderr.write('Step #5: start sender\n')
        self.policy = Policy(self.train)
        self.policy.set_sample_action(self.sample_action)
        self.sender = Sender('192.168.42.222', self.port)
        self.sender.set_policy(self.policy)
        if not self.train:
            self.sender.set_run_time(Config.run_time * 1000)  # ms
            self.sender.policy.set_perf_client(self.perf_client)

        # Update env and corresponding traffic shape in this episode
        self.__update_env_tpg()
        # sys.stderr.write('env reset done\n')
        # sys.stderr.flush()
        return 0

    def rollout(self):
        """Run sender in env, get final reward of an episode, reset sender."""

        # sys.stderr.write('Obtaining an episode from environment...\n')
        ret = self.sender.run()
        return ret

    def cleanup(self):
        if self.expert_client:
            self.expert_client.cleanup()
            # can not set to None

        if self.perf_client:
            self.perf_client.cleanup()
            # can not set to None

        if self.expert_server:
            self.expert_server.kill()
            self.expert_server = None

        if self.perf_server:
            self.perf_client.kill()
            self.perf_server = None

        if self.emulator:
            self.emulator.stop_all()
            self.emulator = None

        if self.sender:
            self.sender.cleanup()
            self.sender = None
