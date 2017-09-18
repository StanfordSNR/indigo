#!/usr/bin/env python

import sys
import argparse
from subprocess import Popen


def add_instances(args):
    num_instances = args.num
    start_num = args.start_num
    prefix = args.prefix
    zone = args.zone
    machine_type = args.type
    img = args.image
    disk_size = args.disk_size

    general_cmd = ('gcloud beta compute --project "edgect-1155" '
                   'instances create "%s%d" --zone "%s" --machine-type "%s" '
                   '--network "default" --maintenance-policy "MIGRATE" '
                   '--service-account "489191239473-compute@developer.gserviceaccount.com" '
                   '--scopes '
                   '"https://www.googleapis.com/auth/devstorage.read_only",'
                   '"https://www.googleapis.com/auth/logging.write",'
                   '"https://www.googleapis.com/auth/monitoring.write",'
                   '"https://www.googleapis.com/auth/servicecontrol",'
                   '"https://www.googleapis.com/auth/service.management.readonly",'
                   '"https://www.googleapis.com/auth/trace.append" '
                   '--min-cpu-platform "Automatic" --image "%s" '
                   '--image-project "edgect-1155" --boot-disk-size "%d" '
                   '--boot-disk-type "pd-standard" '
                   '--boot-disk-device-name "%s%d"')

    procs = []

    for i in xrange(start_num, start_num + num_instances):
        cmd = general_cmd % (prefix, i, zone, machine_type,
                             img, disk_size, prefix, i)
        procs.append(Popen(cmd, shell=True))

    for proc in procs:
        proc.wait()


def change_machine_type(args):
    prefix = args.prefix
    nums = args.nums.split()
    new_type = args.type
    zone = args.zone
    cmd = ('gcloud compute instances set-machine-type "%s" '
           '--machine-type "%s" --zone "%s"')
    procs = []
    for num in nums:
        instance_name = prefix + num
        proc = Popen(cmd % (instance_name, new_type, zone), shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='commands', dest='command')

    add_instance = subparsers.add_parser('add', help='Create instances')
    change_type = subparsers.add_parser('change_type', help='Change inst type')

    add_instance.add_argument(
        '--num', metavar='N', type=int, default=1,
        help='number of instances to create (default: 1)')
    add_instance.add_argument(
        '--start-num', metavar='N', type=int, default=1,
        help='starting prefix to use (default: 1)')
    add_instance.add_argument(
        '--prefix', default='worker-',
        help='prefix for each instance (default: worker-)')
    add_instance.add_argument(
        '--zone', metavar='ZONE', default='us-east1-c',
        help='zone to create in (default: us-east1-c)')
    add_instance.add_argument(
        '--type', metavar='TYPE', default='n1-standard-2',
        help='machine type to use (default: n1-standard-2, 2 cores, 7.5 GB)')
    add_instance.add_argument(
        '--image', metavar='IMG', default='indigo-cpu',
        help='disk img to use (indigo-cpu)')
    add_instance.add_argument(
        '--disk-size', metavar='SIZE', type=int, default='10',
        help='disk size (default: 10 GB)')

    change_type.add_argument(
        '--prefix', metavar='PREFIX', required=True, help='instance prefix')
    change_type.add_argument(
        '--nums', metavar='N1 N2...', required=True, help='instance numbers')
    change_type.add_argument(
        '--type', metavar='TYPE', required=True, help='machine type')
    change_type.add_argument(
        '--zone', metavar='ZONE', default='us-east1-c',
        help='zone (default: us-east1-c)')

    args = parser.parse_args()

    if args.command == 'add':
        add_instances(args)
    elif args.command == 'change_type':
        change_machine_type(args)


if __name__ == '__main__':
    main()
