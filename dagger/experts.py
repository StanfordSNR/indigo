from env.sender import Sender
from helpers.helpers import apply_op
import threading
import sys


def action_error(actions, idx, cwnd, target):
    """ Returns the absolute difference between the target and an action
    applied to the cwnd.
    The action is [op, val] located at actions[idx].
    """
    op = actions[idx][0]
    val = actions[idx][1]
    return abs(apply_op(op, cwnd, val) - target)


def get_best_action(actions, cwnd, target):
    """ Returns the best action by finding the action that leads to the
    closest resulting cwnd to target.
    """
    return min(actions,
               key=lambda idx: action_error(actions, idx, cwnd, target))


class NaiveDaggerExpert(object):
    """ Naive modified LEDBAT implementation """

    def __init__(self):
        self.base_delay = float("inf")
        self.target = 100.0
        self.gain = 1.0

    def sample_action(self, state, cwnd):
        ewma_delay = state      # assume this is the state
        self.base_delay = min(self.base_delay, ewma_delay)
        queuing_delay = ewma_delay - self.base_delay
        off_target = self.target - queuing_delay
        cwnd_inc = self.gain * off_target / cwnd
        target_cwnd = cwnd + cwnd_inc

        # Gets the action that gives the resulting cwnd closest to the
        # expert target cwnd.
        action = get_best_action(Sender.action_mapping, cwnd, target_cwnd)
        return action


class TrueDaggerExpert(object):
    """ Ground truth expert policy """

    def __init__(self, env):
        assert hasattr(env, 'best_cwnd'), ('Using true dagger expert but not '
                                           'given a best cwnd when creating '
                                           'the environment in worker.py.')
        self.best_cwnd = env.best_cwnd
        self.num_flows = env.num_flows


    def change_num_flows(self, interval_num):
        self.num_flows = self.flow_intervals[interval_num][1]
        print 'change flows to %s' % self.num_flows
        sys.stdout.flush()
        if interval_num < len(self.flow_intervals) - 1:
            next_intrvl = interval_num + 1
            next_intrvl_ts = self.flow_intervals[next_intrvl][0]
            print 'waiting for %s seconds to change flows' % next_intrvl_ts
            sys.stdout.flush()
            threading.Timer(next_intrvl_ts,
                            self.change_num_flows, args=[next_intrvl]).start()


    def prepare_flow_intervals(self, flow_intervals):
        self.flow_intervals = flow_intervals
        self.change_num_flows(0)


    def sample_action(self, cwnd):
        # Gets the action that gives the resulting cwnd closest to the
        # best cwnd.
        best_cwnd = max(self.best_cwnd / self.num_flows, 1)
        action = get_best_action(Sender.action_mapping, cwnd, best_cwnd)
        return action
