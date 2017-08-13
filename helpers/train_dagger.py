#!/usr/bin/env python
import sys
import argparse
from subprocess import check_call, check_output


def main():
    """ Runs a sequence of commands to perform DAgger training.
    By default, commits and amends HEAD, updates all machines to latest
    GIT version, and runs the dagger training command.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--username', default='jestinm',
            help='username used in ssh (default: jestinm)')
    parser.add_argument(
            '--rlcc-dir', default='~/RLCC',
            help='path to RLCC/ (default: ~/RLCC)')
    parser.add_argument(
            '--table', default='TABLE',
            help='table for my_gce_helper.py (default: TABLE)')
    parser.add_argument(
            '--commit', default='HEAD',
            help='commit to use on all machines (default: HEAD)')

    args = parser.parse_args()

    gce_helper_cmd = ('%s/helpers/my_gce_helper.py --table %s'
                      % (args.rlcc_dir, args.table))
    gce_helper_out = check_output(gce_helper_cmd, shell=True).split('\n')
    train_cmd = gce_helper_out[0]
    remote_ip = gce_helper_out[1]

    assistant_cmd = ('%s/helpers/my_assistant.py --remote=%s --username=%s '
                     '--rlcc-dir=%s '
                     % (args.rlcc_dir, remote_ip,
                        args.username, args.rlcc_dir))

    if args.commit == 'HEAD':
        check_call(assistant_cmd + 'git_pull', shell=True)
    else:
        checkout_cmd = '"git_checkout %s"' % args.commit
        check_call(assistant_cmd + checkout_cmd, shell=True)

    check_call(train_cmd, shell=True)


if __name__ == '__main__':
    main()
