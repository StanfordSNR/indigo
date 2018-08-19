#!/usr/bin/env python

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


import sys
import signal
from subprocess import call


def signal_handler(signum, frame):
    pass

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# kill mahimahi shells
pkill_cmds = ['pkill -f mm-delay', 'pkill -f mm-link']
# kill all scripts in the directory specified by the first argument
if len(sys.argv) == 2:
    pkill_cmds.append('pkill -f %s' % sys.argv[1])

for cmd in pkill_cmds:
    sys.stderr.write('$ %s\n' % cmd)
    call(cmd, shell=True)
