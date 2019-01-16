#!/usr/bin/env python

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


import sys
import signal

import context  # noqa # pylint: disable=unused-import
from helpers.subprocess_wrappers import Popen


def signal_handler(signum, frame):
    pass


def main():
    # prevent itself from being killed by accident
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    pkill_cmds = []
    # kill all scripts in the directory specified by the first argument
    if len(sys.argv) == 2:
        pkill_cmds.append('pkill -f %s' % sys.argv[1])

    pkill_cmds.append('pkill -f expert_server')
    pkill_cmds.append('mn -c >/dev/null 2>&1')
    pkill_cmds.append('ip link delete veth1 >/dev/null 2>&1')

    procs = []

    for cmd in pkill_cmds:
        procs.append(Popen(cmd, shell=True))

    for proc in procs:
        proc.communicate()


if __name__ == '__main__':
    main()
