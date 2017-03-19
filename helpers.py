import time


def curr_ts_ms():
    return int(time.time() * 1000)


def time_filename():
    return time.strftime('%Y%m%d-%H%M%S')
