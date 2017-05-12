#!/usr/bin/env python

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
