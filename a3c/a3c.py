import sys
import numpy as np
import tensorflow as tf


class A3C(object):
    def __init__(self, state_dim, action_cnt, task_index, session,
                 training=False, save_vars=None, restore_vars=None,
                 debug=False):
        self.state_dim = state_dim
        self.action_cnt = action_cnt
        self.task_index = task_index
        self.session = session
        self.training = training
        self.save_vars = save_vars
        self.restore_vars = restore_vars
        self.debug = debug

        self.device = '/job:worker/task:%s' % task_index

    def update_model(self, state_buf, action_buf, reward):
        sys.stderr.write('Running update_model...\n')

    def sample_action(self, state):
        return np.random.randint(0, self.action_cnt)
