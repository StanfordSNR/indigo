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
import numpy as np
import tensorflow as tf

import context
from policy import Policy
from models import DaggerLSTM
from experts import ExpertClient
from helpers.utils import make_sure_path_exists, one_hot, Config, Status


class DaggerWorker(object):
    def __init__(self, cluster, server, task_idx, env):
        self.cluster = cluster

# private
    def __create_tf_graph(self):
        pass

# public
    def run(self):
        pass

    def cleanup(self):
        pass
