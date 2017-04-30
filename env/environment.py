import os
import sys
import signal
import project_root
from os import path
from sender import Sender
from subprocess import Popen
from helpers.helpers import get_open_udp_port


class Environment(object):
    def __init__(self, mahimahi_cmd):
        self.mahimahi_cmd = mahimahi_cmd

        # variables below will be filled in during setup
        self.state_dim = None
        self.action_cnt = None
        self.sender = None
        self.receiver = None

    def setup(self):
        """Must be called immediately after initializing."""

        self.port = get_open_udp_port()

        # start sender as an instance of Sender class
        sys.stderr.write('Starting sender...\n')
        self.sender = Sender(self.port, train=True)
        self.state_dim = self.sender.state_dim
        self.action_cnt = self.sender.action_cnt

        # start receiver in a subprocess
        sys.stderr.write('Starting receiver...\n')
        receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, self.port)
        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, recv_cmd)
        sys.stderr.write('$ %s\n' % cmd)
        self.receiver = Popen(cmd, preexec_fn=os.setsid, shell=True)

        # sender completes the handshake sent from receiver
        self.sender.handshake()

    def set_sample_action(self, sample_action):
        """Set the sender's policy. Must be called before run()."""

        self.sender.set_sample_action(sample_action)

    def rollout(self):
        """Run sender in env, get an episode of experience and reset sender."""

        sys.stderr.write('Getting an episode from environment...\n')

        self.sender.run()
        rollout = self.sender.get_experience()
        self.sender.reset()

        return rollout

    def cleanup(self):
        if self.sender:
            self.sender.cleanup()

        if self.receiver:
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)

        sys.stderr.write('\nEnvironment cleaned up.\n')
