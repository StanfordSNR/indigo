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
        self.sender = None
        self.receiver = None

    def setup(self):
        """Must be called immediately after initializing.
        """

        self.port = get_open_udp_port()

        # start sender in a separate process
        sys.stderr.write('Starting sender...\n')
        self.sender = Sender(self.port, train=True)

        # start receiver in another process
        sys.stderr.write('Starting receiver...\n')
        receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (receiver_src, self.port)
        cmd = "%s -- sh -c '%s'" % (self.mahimahi_cmd, recv_cmd)
        sys.stderr.write('$ %s\n' % cmd)
        self.receiver = Popen(cmd, preexec_fn=os.setsid, shell=True)

        # wait until the handshake between sender and receiver completes
        self.sender.handshake()

        self.state_dim = self.sender.state_dim
        self.action_cnt = self.sender.action_cnt

    def set_sample_action(self, sample_action):
        self.sender.set_sample_action(sample_action)

    def run(self):
        self.sender.run()

    def get_experience(self):
        return self.sender.get_experience()

    def reset(self):
        self.sender.reset()

    def cleanup(self):
        sys.stderr.write('\nCleaning up environment...\n')

        if self.sender:
            self.sender.cleanup()

        if self.receiver:
            try:
                os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
            except OSError as e:
                sys.stderr.write('%s\n' % e)
