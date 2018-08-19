#!/usr/bin/env python

# Copyright 2018 Francis Y. Yan, Jestin Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import sys
import argparse
from subprocess import check_call, check_output


def main():
    """ Runs a sequence of commands to perform DAgger training. """
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--username', default='francisyyan',
            help='username used in ssh (default: francisyyan)')
    parser.add_argument(
            '--rlcc-dir', default='~/RLCC',
            help='path to RLCC/ (default: ~/RLCC)')
    parser.add_argument(
            '--table', default='TABLE',
            help='table for my_gce_helper.py (default: TABLE)')
    parser.add_argument(
            '--git-push', action='store_true',
            help='git force push and amend latest commit (default: False)')
    parser.add_argument(
            '--git-pull', action='store_true',
            help='whether to do a git pull from all workers (default: False)')
    parser.add_argument(
            '--commit', default='master',
            help='commit for git-pull (default: master)')

    args = parser.parse_args()

    gce_helper_cmd = ('%s/helpers/my_gce_helper.py --table %s --username %s'
                      % (args.rlcc_dir, args.table, args.username))
    gce_helper_out = check_output(gce_helper_cmd, shell=True).split('\n')
    train_cmd = gce_helper_out[0]
    remote_ip = gce_helper_out[1]

    assistant_cmd = ('%s/helpers/assistant.py --remote=%s --username=%s '
                     '--rlcc-dir=%s '
                     % (args.rlcc_dir, remote_ip,
                        args.username, args.rlcc_dir))

    if args.git_push:
        check_call('git add -A && '
                   'git commit --amend --no-edit && '
                   'git push -f', shell=True)

    check_call(assistant_cmd + '--commit=%s git_checkout' % (args.commit), shell=True)

    if args.git_pull:
        check_call(assistant_cmd + 'git_pull', shell=True)

    check_call(train_cmd, shell=True)


if __name__ == '__main__':
    main()
