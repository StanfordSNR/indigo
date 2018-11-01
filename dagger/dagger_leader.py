# Copyright 2018 Francis Y. Yan, Jestin Ma
# Copyright 2018 Yiyang Shao, Wei Wang (Huawei Technologies)
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


import sys
import time
import datetime
import socket
import numpy as np
import tensorflow as tf
from subprocess import check_output
from os import path

import context
from policy import Policy
from models import DaggerLSTM
from helpers.utils import (
    make_sure_path_exists, timestamp_ms, softmax, Config, Status)


class DaggerLeader(object):
    def __init__(self, cluster, server, worker_tasks):
        self.cluster = cluster

        self.state_dim = Policy.state_dim
        self.action_cnt = Policy.action_cnt

        # augmented state space: state and previous action (one-hot vector)
        self.aug_state_dim = self.state_dim + self.action_cnt

        # create Tensorflow dataflow graph
        self.__create_tf_graph()


# private
    def __create_tf_graph(self):
        # create the global model on the PS server(s)
        # place variables on GPU if available, and on CPU otherwise
        with tf.device(tf.train.replica_device_setter(cluster=self.cluster)):
            self.global_model = DaggerLSTM(state_dim=self.aug_state_dim,
                                           action_cnt=self.action_cnt)

# public
    def run(self):
        pass

    def cleanup(self):
        pass
