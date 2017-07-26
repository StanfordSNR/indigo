#!/bin/bash
~/RLCC/helpers/my_assistant.py --remote=10.240.0.37,10.240.0.38,10.240.0.39,10.240.0.40,10.240.0.41 cleanup
~/RLCC/dagger/train.py --username jestinm --rlcc-dir /home/jestinm/RLCC --ps-hosts 10.240.0.37:15000 --worker-hosts 10.240.0.38:16000,10.240.0.39:16001,10.240.0.40:16002,10.240.0.41:16003
