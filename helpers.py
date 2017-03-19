import time


def curr_ts_ms():
    return int(time.time() * 1000)


def time_filename():
    return time.strftime('%Y%m%d-%H%M%S')


def test_sender_rl_params():
    def test_sample_action(state):
        time.sleep(1)
        return 1

    rl_params = {
        'state_dim': 10,
        'action_num': 3,
        'max_steps': 10,
        'delay_weight': 0.8,
        'sample_action': test_sample_action,
    }

    return rl_params
