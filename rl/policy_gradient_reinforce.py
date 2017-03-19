import time


class PolicyGradientReinforce(object):
    def __init__(self, **params):
        self.session = params['session']
        self.optimizer = params['optimizer']
        self.state_dim = params['state_dim']
        self.action_cnt = params['action_cnt']

    def policy_network(self, states):
        return 1

    def sample_action(self, states):
        time.sleep(1)
        return self.policy_network(states)

    def store_experience(self, experience):
        return

    def update_model(self):
        return
