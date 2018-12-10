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


import os
import argparse

import tensorflow as tf
from subprocess import call

import context
from dagger.sender import LSTMExecuter, Sender
from dagger.policy import Policy
from env.mininet_env import MininetEnv
from helpers.utils import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', action='store')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    env = MininetEnv(Config.total_env_set_test, Config.total_tpg_set_test, False)

    try:
        while not env.is_all_tasks_done():
            
            lstm = LSTMExecuter(state_dim=Policy.state_dim,
                             action_cnt=Policy.action_cnt,
                             restore_vars=args.model_path)
            env.set_sample_action(lstm.sample_action)

            if env.reset() != -1:
                env.rollout()

            tf.reset_default_graph()
    except KeyboardInterrupt:
        pass
    finally:
        env.cleanup()
        call('mn -c', shell=True)
        return


if __name__ == '__main__':
    main()
