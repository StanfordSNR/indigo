#!/usr/bin/env python
import sys
import argparse
from subprocess import check_call, check_output


def main():
    """ Runs a sequence of commands to perform DAgger training.
    Removes the past run's history from all workers (saves to the current
    machine if wanted, adds all edits to the HEAD
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
            '--rm-history', action='store_false',
            help='delete history from all workers (default: True)')
    parser.add_argument(
            '--git-push', action='store_false',
            help='git push and amend on all workers (default: True)')
    parser.add_argument(
            '--git-pull', action='store_false',
            help='whether to do a git pull from all workers (default: True)')
        
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

    if args.rm_history:
        check_call(assistant_cmd + 'rm_history', shell=True)

    if args.git_push:
        check_call('git add -A && '
                   'git commit --amend --no-edit && '
                   'git push -f', shell=True)

    if args.git_pull:
        check_call(assistant_cmd + 'git_pull', shell=True)

    check_call(train_cmd, shell=True)


if __name__ == '__main__':
    main()
