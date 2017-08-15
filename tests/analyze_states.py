#!/usr/bin/python

import numpy as np
import argparse

NUM_STATES = 7

def main():
    """ open the states log and compute various statistics """

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--states-log", default='/home/jestinm/RLCC/tests/states.log')

    args = parser.parse_args()
    states_log = open(args.states_log)
    state_buf = [[] for i in xrange(NUM_STATES)]

    while True:
        line = states_log.readline()
        if not line:
            break

        new_data = map(float, line.split())
        if len(new_data) != NUM_STATES:
            continue

        for i in xrange(NUM_STATES):
            state_buf[i].append(new_data[i])

    states_log.close()

    print '99th percentiles: %s\n' % str(np.percentile(state_buf, 99, axis=1))
    print '1st percentiles: %s\n' % str(np.percentile(state_buf, 1, axis=1))
    print 'mean: %s\n' % str(np.mean(state_buf, axis=1))
    print 'stddev: %s\n' % str(np.std(state_buf, axis=1))

if __name__ == '__main__':
    main()
