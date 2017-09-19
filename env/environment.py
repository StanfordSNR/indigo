import os
from os import path
import sys
import signal
from subprocess import Popen, PIPE
from sender import Sender
import project_root
from helpers.helpers import get_open_udp_port


class Environment(object):
    def __init__(self, mahimahi_cmd, num_flows,
                 start_worker, end_worker, best_cwnd=None):
        self.mahimahi_cmd = mahimahi_cmd
        self.state_dim = Sender.state_dim
        self.action_cnt = Sender.action_cnt
        self.start_worker = start_worker
        self.end_worker = end_worker
        self.best_cwnd = best_cwnd
        self.num_flows = num_flows

        # variables below will be filled in during setup
        self.receivers = None

    def reset(self, ports):
        """Must be called once before running rollout()."""

        self.cleanup_receivers()

        # start receivers in a mahimahi subprocess (self.receivers)
        sys.stderr.write('Starting receivers for workers %s-%s\n' %
                        (self.start_worker, self.end_worker))

        receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        self.receivers = Popen(self.mahimahi_cmd, stdin=PIPE,
                               preexec_fn=os.setsid, shell=True)
        recv_cmds = []
        for port in ports:
            cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, port)
            sys.stderr.write('$ %s\n' % cmd)
            full_cmd = "sh -c '%s' &" % cmd
            self.receivers.stdin.write(full_cmd + '\n')
            self.receivers.stdin.flush()

    def cleanup_receivers(self):
        if self.receivers:
            try:
                os.killpg(os.getpgid(self.receivers.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
            finally:
                    self.receivers = None
