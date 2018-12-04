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
import ast
import shutil
import argparse
import ConfigParser
import tensorflow as tf
from subprocess import call

import context
from dagger.run_sender import Learner
from dagger.experts import PerfClient
from env.mininet_env import MininetEnv
from env.sender import Sender


def get_mininet_env_param():
    total_tp_set = []
    total_env_set = []

    cfg = ConfigParser.ConfigParser()
    cfg_path = os.path.join(context.base_dir, 'config.ini')
    cfg.read(cfg_path)

    test_env = cfg.options('test_env')
    for opt in test_env:
        env_param, tp_set_param = ast.literal_eval(cfg.get('test_env', opt))
        total_tp_set.append(ast.literal_eval(cfg.get('global', tp_set_param)))
        total_env_set.append(ast.literal_eval(cfg.get('global', env_param)))

    return total_tp_set, total_env_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', action='store')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # set up dir for log
    perf_log_path = os.path.join(context.base_dir, 'tests', 'perf_log')
    rtt_loss_path = os.path.join(context.base_dir, 'tests', 'rtt_loss')
    if os.path.exists(perf_log_path):
        shutil.rmtree(perf_log_path)
    if os.path.exists(rtt_loss_path):
        shutil.rmtree(rtt_loss_path)
    os.makedirs(perf_log_path)
    os.makedirs(rtt_loss_path)

    model_path = args.model_path

    pc = PerfClient()

    total_tpg_set, total_env_set = get_mininet_env_param()
    env = MininetEnv(total_env_set, total_tpg_set, False)
    env.set_expert(pc)

    try:
        while True:
            learner = Learner(
                state_dim=Sender.state_dim,
                action_cnt=Sender.action_cnt,
                restore_vars=model_path)
            env.set_sample_action(learner.sample_action)

            env.reset()
            env.rollout()

            tf.reset_default_graph()

            if env.is_all_tasks_done():
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.cleanup()
        call('mn -c', shell=True)
        call('pkill -f perf_server', shell=True)
        return


if __name__ == '__main__':
    main()
