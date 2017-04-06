import time
import numpy as np


def curr_ts_ms():
    return int(time.time() * 1000)


class RingBuffer(object):
    def __init__(self, length):
        self.data = np.zeros(length)
        self.index = 0

    def append(self, x):
        self.data[self.index] = x
        self.index = (self.index + 1) % self.data.size

    def get(self):
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]

    def reset(self):
        self.data.fill(0)
        self.index = 0


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError()
