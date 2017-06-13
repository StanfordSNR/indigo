#!/usr/bin/env python

import numpy as np
import project_root
from helpers.helpers import RingBuffer, MeanVarHistory


def test_ring_buffer():
    buf = RingBuffer(5)

    buf.append(1)
    buf.append(2)
    assert np.array_equal(buf.get(), np.array([1, 2]))
    assert buf.get().dtype == np.float
    buf.reset()

    for i in xrange(1, 7):
        buf.append(i)
    assert np.array_equal(buf.get(), np.array([2, 3, 4, 5, 6]))
    buf.reset()

    assert np.array_equal(buf.get(), np.array([]))

    print 'test_ring_buffer: success'


def test_mean_var_history():
    h = MeanVarHistory()

    h.append([1, 2, 3])
    h.append(np.array([4, 5]))

    correct = np.array([1, 2, 3, 4, 5])
    assert np.isclose(h.get_mean(), np.mean(correct))
    assert np.isclose(h.get_var(), np.var(correct))
    assert np.isclose(h.get_std(), np.std(correct))

    h.reset()

    h.append([1, 1, 1])
    assert np.allclose(h.normalize_copy([2, 3, 4]), np.array([1e5, 2e5, 3e5]))

    h.append([3, 3, 3])
    x = np.array([2.0, 4.0])
    h.normalize_inplace(x)
    assert np.allclose(x, np.array([0.0, 2.0]))

    print 'test_mean_var_history: success'


def main():
    test_ring_buffer()
    test_mean_var_history()


if __name__ == '__main__':
    main()
