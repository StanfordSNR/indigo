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
import time
import commands
import subprocess

times = 10000

now = time.time()
for i in xrange(times):
    os.system('tc qdisc change dev h1-eth0 root fq maxrate 8m')
cmd1 = time.time() - now
print('os.system exec time: {}'.format(cmd1))

now = time.time()
for i in xrange(times):
    os.popen('tc qdisc change dev h1-eth0 root fq maxrate 8000000')
cmd2 = time.time() - now
print('os.popen exec time: {}'.format(cmd2))

now = time.time()
for i in xrange(times):
    commands.getstatusoutput('tc qdisc change dev h1-eth0 root fq maxrate 8000000')
cmd3 = time.time() - now
print('commands.getstatusoutput exec time: {}'.format(cmd3))

now = time.time()
for i in xrange(times):
    subprocess.call('tc qdisc change dev h1-eth0 root fq maxrate 8000000', shell=True)
cmd4 = time.time() - now
print('subprocess.call exec time: {}'.format(cmd4))

now = time.time()
for i in xrange(times):
    subprocess.Popen('tc qdisc change dev h1-eth0 root fq maxrate 8000000', shell=True)
cmd5 = time.time() - now
print('subprocess.Popen exec time: {}'.format(cmd5))

now = time.time()
for i in xrange(times):
    subprocess.check_output(['tc', 'qdisc', 'change', 'dev', 'h1-eth0', 'root', 'fq', 'maxrate', '8000000'])
cmd6 = time.time() - now
print('subprocess.check_output exec time: {}'.format(cmd6))
