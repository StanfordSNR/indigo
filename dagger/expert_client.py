#!/usr/bin/env python

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

import socket
import sys

from policy import Policy


def action_error(action, cwnd, target):
    """ Returns the absolute difference between the target and an action
    applied to the cwnd.
    The action is [op, val] located at actions[idx].
    """
    op, val = action
    return abs(op(cwnd, val) - target)


def get_best_action(actions, cwnd, target):
    """ Returns the best action by finding the action that leads to the
    closest resulting cwnd to target.
    """
    return actions.index(min(actions,
                             key=lambda action: action_error(action, cwnd, target)))


class NaiveDaggerExpert(object):
    """ Naive modified LEDBAT implementation """

    def __init__(self):
        self.base_delay = float("inf")
        self.target = 100.0
        self.gain = 1.0

    def sample_action(self, state, cwnd):
        ewma_delay = state  # assume this is the state
        self.base_delay = min(self.base_delay, ewma_delay)
        queuing_delay = ewma_delay - self.base_delay
        off_target = self.target - queuing_delay
        cwnd_inc = self.gain * off_target / cwnd
        target_cwnd = cwnd + cwnd_inc

        # Gets the action that gives the resulting cwnd closest to the
        # expert target cwnd.
        action = get_best_action(Policy.action_mapping, cwnd, target_cwnd)
        return action


class ExpertClient(object):
    """ Get ground truth expert policy from expert server by socket.
        This is client"""

    def __init__(self):
        self.socket = None

    def cleanup(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def connect_expert_server(self, port):
        self.address = ('0.0.0.0', port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect(self.address)
        except socket.error:
            sys.stderr.write('connect to expert socket error\n')
            return -1
        return 0

    def sample_action(self, cwnd):
        # Gets the action that gives the resulting cwnd closest to the
        # best cwnd.

        self.socket.send('Current best cwnd?')
        msg = self.socket.recv(512)

        try:
            best_cwnd = float(msg)
        except ValueError:
            sys.stderr.write('Expert server returns invalid best cwnd {} \n'.format(msg))
            self.best_cwnd = cwnd
            return Policy.action_cnt / 2

        self.best_cwnd = best_cwnd
        action = get_best_action(Policy.action_mapping, cwnd, self.best_cwnd)

        return action
