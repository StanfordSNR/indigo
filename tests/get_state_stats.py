#!/usr/bin/python

import project_root
import argparse
import os
import signal
from subprocess import check_call, check_output, Popen, PIPE
from helpers.helpers import make_sure_path_exists

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--rlcc-dir', default='/home/jestinm/RLCC',
            help='path to RLCC/ (default: /home/jestinm/RLCC)')
    parser.add_argument(
            '--trace-folder', default='/tmp/traces',
            help='path to bandwidth traces (default: /tmp/traces)')
    parser.add_argument(
            "--delays",
            help="one way propagation delays in ms")
    parser.add_argument(
            "--bandwidths",
            help="(link rates)")

    args = parser.parse_args()

    # Make sure the bandwidth traces exist in trace folder
    make_sure_path_exists(args.trace_folder)

    for bandwidth in args.bandwidths.split():
        check_call('%s/helpers/generate_trace.py --bandwidth %s --output-dir %s'
                    % (args.rlcc_dir, bandwidth, args.trace_folder),
                    shell=True)

    # for each combination of bandwidth & trace, run the sender and receiver
    for bandwidth in args.bandwidths.split():
        for delay in args.delays.split():
            sender = Popen('%s/tests/test_sender.py 12345 >> %s/tests/states.log'
                            % (args.rlcc_dir, args.rlcc_dir),
                            shell=True)

            recv = Popen("mm-delay %s mm-link %s/%smbps.trace %s/%smbps.trace -- "
                         "sh -c '%s/env/run_receiver.py $MAHIMAHI_BASE 12345'"
                          % (delay, args.trace_folder, bandwidth,
                             args.trace_folder, bandwidth, args.rlcc_dir),
                          shell=True,
                          preexec_fn = os.setsid)

            sender.communicate()
            os.killpg(os.getpgid(recv.pid), signal.SIGTERM)

    check_call('%s/tests/analyze_states.py --states-log=%s/tests/states.log'
                % (args.rlcc_dir, args.rlcc_dir),
                shell=True)


if __name__ == '__main__':
    main()
