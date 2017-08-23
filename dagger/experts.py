import project_root
from env.sender import Sender
from helpers.helpers import apply_op


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

    def sample_action(self, cwnd):
        # Gets the action that gives the resulting cwnd closest to the
        # best cwnd.
        action = get_best_action(Sender.action_mapping, cwnd, self.best_cwnd)
        return action


class DaggerCoach(object):
    def __init__(self, env):
        assert hasattr(env, 'best_cwnd')
        assert hasattr(env, 'action_cnt')

        self.best_cwnd = env.best_cwnd
        self.action_cnt = env.action_cnt

    def sample_action(self, eps, cwnd, action_probs=None):
        if eps == 0:
            lambd = 0.0
            action_probs = [0.0 for _ in xrange(self.action_cnt)]
        else:
            assert action_probs is not None

            if eps == 1:
                lambd = 1.0
            else:
                lambd = 0.8 ** (eps - 1)

        max_obj = None
        max_action_idx = None

        for action_idx in xrange(self.action_cnt):
            immediate_loss = self.immediate_loss(cwnd, action_idx)
            obj = lambd * action_probs[action_idx] - immediate_loss

            if max_obj is None or obj > max_obj:
                max_obj = obj
                max_action_idx = action_idx

        return max_action_idx

    def immediate_loss(self, cwnd, action_idx):
        op, val = Sender.action_mapping[action_idx]
        abs_diff = abs(apply_op(op, cwnd, val) - self.best_cwnd)

        return 1.0 * abs_diff / self.best_cwnd
