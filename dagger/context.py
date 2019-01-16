#!/usr/bin/env python

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(base_dir)
