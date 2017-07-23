#!/bin/bash
git add -A
git commit --amend 
git push -f
REMOTE=${1:-local}
if [ $REMOTE = "remote" ]; then
    ./my_assistant.py --remote=10.240.0.37,10.240.0.38,10.240.0.39,10.240.0.40,10.240.0.41 git_pull
fi
