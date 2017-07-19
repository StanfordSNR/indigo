#!/bin/sh

./my_assistant.py --remote=$1 cleanup
./my_assistant.py --remote=$1 git_pull
./my_assistant.py --remote=$1 inc_rmem
./setup.py --remote=$1
