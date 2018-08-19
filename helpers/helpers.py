# Copyright 2018 Francis Y. Yan, Jestin Ma
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


import os
import time
import errno
import select
import socket
import numpy as np
import operator


READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS

math_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.div,
}


def apply_op(op, op1, op2):
    return math_ops[op](op1, op2)


def curr_ts_ms():
    if not hasattr(curr_ts_ms, 'epoch'):
        curr_ts_ms.epoch = time.time()

    return int((time.time() - curr_ts_ms.epoch) * 1000)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_open_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def normalize(state):
    return [state[0] / 200.0, state[1] / 200.0,
            state[2] / 200.0, state[3] / 5000.0]


def one_hot(action, action_cnt):
    ret = [0.0] * action_cnt
    ret[action] = 1.0
    return ret


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class RingBuffer(object):
    def __init__(self, length):
        self.full_len = length
        self.real_len = 0
        self.index = 0
        self.data = np.zeros(length)

    def append(self, x):
        self.data[self.index] = x
        self.index = (self.index + 1) % self.full_len
        if self.real_len < self.full_len:
            self.real_len += 1

    def get(self):
        idx = (self.index - self.real_len +
               np.arange(self.real_len)) % self.full_len
        return self.data[idx]

    def reset(self):
        self.real_len = 0
        self.index = 0
        self.data.fill(0)


class MeanVarHistory(object):
    def __init__(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0

    def append(self, x):
        """Append x to history.

        Args:
            x: a list or numpy array.
        """
        # x: a list or numpy array
        length_new = self.length + len(x)
        ratio_old = float(self.length) / length_new
        ratio_new = float(len(x)) / length_new

        self.length = length_new
        self.mean = self.mean * ratio_old + np.mean(x) * ratio_new
        self.square_mean = (self.square_mean * ratio_old +
                            np.mean(np.square(x)) * ratio_new)
        self.var = self.square_mean - np.square(self.mean)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.var if self.var > 0 else 1e-10

    def get_std(self):
        return np.sqrt(self.get_var())

    def normalize_copy(self, x):
        """Normalize x and returns a copy.

        Args:
            x: a list or numpy array.
        """
        return [(v - self.mean) / self.get_std() for v in x]

    def normalize_inplace(self, x):
        """Normalize x in place.

        Args:
            x: a numpy array with float dtype.
        """
        x -= self.mean
        x /= self.get_std()

    def reset(self):
        self.length = 0
        self.mean = 0.0
        self.square_mean = 0.0
        self.var = 0.0
