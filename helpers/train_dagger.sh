#!/bin/bash
ssh jestinm@10.240.0.38 -- rm ~/RLCC/history
~/RLCC/a3c/train.py --username jestinm --rlcc-dir /home/jestinm/RLCC --ps-hosts 10.240.0.37:15000 --worker-hosts 10.240.0.38:16000,10.240.0.39:16001,10.240.0.40:16002,10.240.0.41:16003 --dagger
