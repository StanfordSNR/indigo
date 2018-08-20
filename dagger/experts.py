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


import socket
from env.sender import Sender
from helpers.helpers import apply_op


def action_error(actions, idx, cwnd, target):
    """ Returns the absolute difference between the target and an action
    applied to the cwnd.
    The action is [op, val] located at actions[idx].
    """
    op = actions[idx][0]
    val = actions[idx][1]
    return abs(apply_op(op, cwnd, val) - target)


def get_best_action(actions, cwnd, target):
    """ Returns the best action by finding the action that leads to the
    closest resulting cwnd to target.
    """
    return min(actions, key=lambda idx: action_error(actions, idx, cwnd, target))


class NaiveDaggerExpert(object):
    """ Naive modified LEDBAT implementation """

    def __init__(self):
        self.base_delay = float("inf")
        self.target = 100.0
        self.gain = 1.0

    def sample_action(self, state, cwnd):
        ewma_delay = state      # assume this is the state
        self.base_delay = min(self.base_delay, ewma_delay)
        queuing_delay = ewma_delay - self.base_delay
        off_target = self.target - queuing_delay
        cwnd_inc = self.gain * off_target / cwnd
        target_cwnd = cwnd + cwnd_inc

        # Gets the action that gives the resulting cwnd closest to the
        # expert target cwnd.
        action = get_best_action(Sender.action_mapping, cwnd, target_cwnd)
        return action


class TrueDaggerExpert(object):
    """ Ground truth expert policy """

    def __init__(self, env):
        assert hasattr(env, 'best_cwnd'), ('Using true dagger expert but not '
                                           'given a best cwnd when creating '
                                           'the environment in worker.py.')
        self.best_cwnd = env.best_cwnd

    def sample_action(self, cwnd):
        # Gets the action that gives the resulting cwnd closest to the
        # best cwnd.
        action = get_best_action(Sender.action_mapping, cwnd, self.best_cwnd)
        return action


class Mininet_Expert(object):
    """ Ground truth expert policy """

    def __init__(self):
        self.socket = None

    def cleanup(self):
        if self.socket:
            self.socket.close()

    def connect_expert_server(self, port):
        self.address = ('0.0.0.0', port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect(self.address)
            print 'expert socket connect success'
        except socket.error:
            print 'connect to expert socket error:'
            return -1
        return 0

    def sample_action(self, cwnd):
        # Gets the action that gives the resulting cwnd closest to the
        # best cwnd.

        self.socket.send('Hello')
        msg = self.socket.recv(512)
        try:
            best_cwnd = float(msg)
        except ValueError:
            return -1

        self.best_cwnd = best_cwnd
        action = get_best_action(Sender.action_mapping, cwnd, self.best_cwnd)
        # print '----best----privous----best action----', self.best_cwnd, cwnd, action
        return action


class Perf_Client(object):

    def __init__(self):
        self.socket = None

    def cleanup(self):
        if self.socket:
            self.socket.close()

    def connect_perf_server(self, port):
        self.address = ('0.0.0.0', port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect(self.address)
            print 'perf server socket connect success'
        except socket.error:
            print 'connect to perf server error:'
            return -1
        return 0

    def upload_cwnd(self, cwnd):
        self.socket.send(str(cwnd))
