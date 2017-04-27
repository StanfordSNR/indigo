import os
import sys
import signal
import multiprocessing
import project_root
from os import path
from sender import Sender
from subprocess import Popen
from helpers.helpers import get_open_udp_port


class Environment(object):
    def __init__(self, mahimahi_cmd):
        self.port = get_open_udp_port()

        # run sender in a separate process
        sender_proc = multiprocessing.Process(target=self.run_sender)

        # run receiver in another process
        sys.stderr.write('Starting receiver...\n')
        run_receiver_src = path.join(
            project_root.DIR, 'env', 'run_receiver.py')
        recv_cmd = 'python %s $MAHIMAHI_BASE %s' % (run_receiver_src, port)
        cmd = "%s -- sh -c '%s'" % (mahimahi_cmd, recv_cmd)
        sys.stderr.write('$ %s\n' % cmd)
        receiver_proc = Popen(cmd, preexec_fn=os.setsid)

        # wait until the handshake between sender and receiver completes
        sender_proc.join()

        self.state_dim = self.sender.state_dim
        self.action_cnt = self.sender.action_cnt

    def run_sender(self):
        sys.stderr.write('Starting sender...\n')
        self.sender = Sender(self.port, train=True)

    def set_sample_action(self, sample_action):
        self.sender.set_sample_action(sample_action)

    def run(self):
        self.sender.run()

    def get_experience(self):
        return self.sender.get_experience()

    def reset(self):
        self.sender.reset()

    def clean_up(self):
        self.sender.clean_up()

        try:
            os.killpg(os.getpgid(self.receiver.pid), signal.SIGTERM)
        except OSError as e:
            sys.stderr.write('%s\n' % e)
