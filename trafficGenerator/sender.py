#!/usr/bin/env python

# Copyright 2018 Huawei Technologies
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


import commands
import os
import socket
import subprocess
import sys
import time


class Function:
    def __init__(self, expression, domain, definition):
        self.func_Mbps = expression
        self.lower_s = float(domain[0])
        self.upper_s = float(domain[1])
        self.definition_s = float(definition)

    def get_pacing_bin(self):
        pacing_bin = []
        curr_s = self.lower_s
        while curr_s < self.upper_s:
            abs_rate_bps = self.func_Mbps(curr_s) * 1000 * 1000
            pacing_bps = int(abs_rate_bps / 8) * 8
            pacing_bin.append(pacing_bps)
            curr_s += self.definition_s
        return pacing_bin


def min_x_max(min_value, x, max_value):
    # return x if min_value < x < max_value
    # else return min_value or max_value
    if min_value < x < max_value:
        return x
    elif x >= max_value:
        return max_value
    else:
        return min_value


def get_func(shape, bound):

    bandwitdh_ratio = 0

    def func_expression_1(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 1.0 + 20.0 * x, 1000.0)

    def func_expression_2(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 200.0 + 20.0 * x, 1000.0)

    def func_expression_3(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 400.0 + 20.0 * x, 1000.0)

    def func_expression_4(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 600.0 + 20.0 * x, 1000.0)

    def func_expression_5(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 800.0 + 20.0 * x, 1000.0)

    def func_expression_6(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 1000.0 - 20.0 * x, 1000.0)

    def func_expression_7(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 800.0 - 20.0 * x, 1000.0)

    def func_expression_8(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 600.0 - 20.0 * x, 1000.0)

    def func_expression_9(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 400.0 - 20.0 * x, 1000.0)

    def func_expression_10(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min_x_max(0.0, 200.0 - 20.0 * x, 1000.0)

    def func_expression_11(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 1000
        else:
            return max(1000-20*(x-5), 100)

    def func_expression_12(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(1+2.0 * x, 100)

    def func_expression_13(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(20+2.0 * x, 100)

    def func_expression_14(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(40+2.0 * x, 100)

    def func_expression_15(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(60+2.0 * x, 100)

    def func_expression_16(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(80+2.0 * x, 100)

    def func_expression_17(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(100 - 2.0 * x, 100)

    def func_expression_18(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(80 - 2.0 * x, 100)

    def func_expression_19(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(60 - 2.0 * x, 100)

    def func_expression_20(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(40 - 2.0 * x, 100)

    def func_expression_21(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(20 - 2.0 * x, 100)

    def func_expression_22(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 100
        else:
            return max(100-2*(x-5), 10)

    def func_expression_23(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0+0.02 * x, 0.9)

    def func_expression_24(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.2 + 0.02 * x, 0.9)

    def func_expression_25(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.4 + 0.02 * x, 0.9)

    def func_expression_26(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.6 + 0.02 * x, 0.9)

    def func_expression_27(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.8 + 0.02 * x, 0.9)

    def func_expression_28(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(1 - 0.02 * x, 0.9)

    def func_expression_29(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.8 - 0.02 * x, 0.9)

    def func_expression_30(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.6 - 0.02 * x, 0.9)

    def func_expression_31(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.4 - 0.02 * x, 0.9)

    def func_expression_32(x):    # unit: Mbps
        x = float(x)             # unit: second
        return min(0.2 - 0.02 * x, 0.9)

    def func_expression_33(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 1
        else:
            return min(1-0.02*(x-5), 0.9)

    def func_expression_50(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 100.0 * x, 999.0)

    def func_expression_51(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 1000.0 - 100.0 * x, 999.0)

    def func_expression_52(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 200
        else:
            return min_x_max(200.0, 200 + 100.0 * (x - 5), 999.0)

    def func_expression_53(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 800
        else:
            return min_x_max(0.0, 800 - 100.0 * (x - 5), 800.0)

    def func_expression_60(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 50.0 * x, 499.0)

    def func_expression_61(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 500.0 - 50.0 * x, 499.0)

    def func_expression_62(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 100
        else:
            return min_x_max(100.0, 100 + 50.0 * (x - 5), 499.0)

    def func_expression_63(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 400
        else:
            return min_x_max(0.0, 400 - 50.0 * (x - 5), 400.0)

    def func_expression_70(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 10.0 * x, 99.0)

    def func_expression_71(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 100.0 - 10.0 * x, 99.0)

    def func_expression_72(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 20
        else:
            return min_x_max(20.0, 20 + 10.0 * (x - 5), 99.0)

    def func_expression_73(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 80
        else:
            return min_x_max(0.0, 80 - 10.0 * (x - 5), 80.0)

    def func_expression_1000(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 10.0 * bandwitdh_ratio * x, 100.0 * bandwitdh_ratio-1)

    def func_expression_1001(x):    # unit: Mbps
        x = float(x)              # unit: second
        return min_x_max(0.0, 100.0 * bandwitdh_ratio - 10.0 * bandwitdh_ratio * x, 100.0 * bandwitdh_ratio-1)

    def func_expression_1002(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 20 * bandwitdh_ratio
        else:
            return min_x_max(20.0 * bandwitdh_ratio, 20 + 10.0 * bandwitdh_ratio * (x - 5), 100.0 * bandwitdh_ratio-1)

    def func_expression_1003(x):    # unit: Mbps
        x = float(x)             # unit: second
        if x <= 5:
            return 80 * bandwitdh_ratio
        else:
            return min_x_max(0.0, 80 * bandwitdh_ratio - 10.0 * bandwitdh_ratio * (x - 5), 80.0 * bandwitdh_ratio)

    def func_expression_101(x):
        return 0

    def func_expression_102(x):
        return 900

    def func_expression_103(x):
        x = float(x)
        return min_x_max(0.0, 1000.0 - 20.0 * x, 1000.0)

    def func_expression_104(x):
        x = float(x)
        return min_x_max(0.0, 20.0 * x, 1000.0)

    def func_expression_105(x):
        x = int(x)
        x = x % 20
        x = int(x / 2)
        if x == 0:
            return 0.0
        elif x == 1:
            return 900.0
        elif x == 2:
            return 100.0
        elif x == 3:
            return 800.0
        elif x == 4:
            return 200.0
        elif x == 5:
            return 700.0
        elif x == 6:
            return 300.0
        elif x == 7:
            return 600.0
        elif x == 8:
            return 400.0
        else:
            return 500.0

    def func_expression_106(x):
        x = float(x)
        if x < 10:
            return 500
        elif x < 20:
            return min_x_max(0.0, 500.0 - 200.0 * (x - 10), 1000.0)
        else:
            return min_x_max(0.0, 200.0 * (x - 20), 1000.0)

    def func_expression_107(x):
        x = float(x)
        if x < 10:
            return 200
        elif x < 20:
            return min_x_max(0.0, 500.0 + 20.0 * (x - 10), 1000.0)
        else:
            return min_x_max(0.0, 1000.0 - 20.0 * (x - 20), 1000.0)

    def func_expression_108(x):
        x = float(x)
        if x < 10:
            return 800
        elif x < 15:
            return min_x_max(0.0, 800.0 - 100.0 * (x - 10), 1000.0)
        elif x < 20:
            return min_x_max(0.0, 300.0 + 120.0 * (x - 15), 1000.0)
        elif x < 25:
            return min_x_max(0.0, 900.0 - 180.0 * (x - 20), 1000.0)
        else:
            return min_x_max(0.0, 200.0 * (x - 25), 1000.0)

    def func_expression_111(x):
        return 0

    def func_expression_112(x):
        return 450

    def func_expression_113(x):
        x = float(x)
        return min_x_max(0.0, 500.0 - 10.0 * x, 500.0)

    def func_expression_114(x):
        x = float(x)
        return min_x_max(0.0, 10.0 * x, 500.0)

    def func_expression_115(x):
        x = int(x)
        x = x % 20
        x = int(x / 2)
        if x == 0:
            return 0.0
        elif x == 1:
            return 450.0
        elif x == 2:
            return 50.0
        elif x == 3:
            return 400.0
        elif x == 4:
            return 100.0
        elif x == 5:
            return 350.0
        elif x == 6:
            return 150.0
        elif x == 7:
            return 300.0
        elif x == 8:
            return 200.0
        else:
            return 250.0

    def func_expression_116(x):
        x = float(x)
        if x < 10:
            return 250
        elif x < 20:
            return min_x_max(0.0, 250.0 - 100.0 * (x - 10), 1000.0)
        else:
            return min_x_max(0.0, 100.0 * (x - 20), 1000.0)

    def func_expression_117(x):
        x = float(x)
        if x < 10:
            return 100
        elif x < 20:
            return min_x_max(0.0, 250.0 + 10.0 * (x - 10), 500.0)
        else:
            return min_x_max(0.0, 500.0 - 10.0 * (x - 20), 500.0)

    def func_expression_118(x):
        x = float(x)
        if x < 10:
            return 400
        elif x < 15:
            return min_x_max(0.0, 400.0 - 50.0 * (x - 10), 500.0)
        elif x < 20:
            return min_x_max(0.0, 150.0 + 60.0 * (x - 15), 500.0)
        elif x < 25:
            return min_x_max(0.0, 450.0 - 90.0 * (x - 20), 500.0)
        else:
            return min_x_max(0.0, 100.0 * (x - 25), 500.0)

    def func_expression_121(x):
        return 0

    def func_expression_122(x):
        return 90

    def func_expression_123(x):
        x = float(x)
        return min_x_max(0.0, 100.0 - 2.0 * x, 100.0)

    def func_expression_124(x):
        x = float(x)
        return min_x_max(0.0, 2.0 * x, 100.0)

    def func_expression_125(x):
        x = int(x)
        x = x % 20
        x = int(x / 2)
        if x == 0:
            return 0.0
        elif x == 1:
            return 90.0
        elif x == 2:
            return 10.0
        elif x == 3:
            return 80.0
        elif x == 4:
            return 20.0
        elif x == 5:
            return 70.0
        elif x == 6:
            return 30.0
        elif x == 7:
            return 60.0
        elif x == 8:
            return 40.0
        else:
            return 50.0

    def func_expression_126(x):
        x = float(x)
        if x < 10:
            return 50
        elif x < 20:
            return min_x_max(0.0, 50.0 - 20.0 * (x - 10), 100.0)
        else:
            return min_x_max(0.0, 20.0 * (x - 20), 100.0)

    def func_expression_127(x):
        x = float(x)
        if x < 10:
            return 20
        elif x < 20:
            return min_x_max(0.0, 50.0 + 2.0 * (x - 10), 100.0)
        else:
            return min_x_max(0.0, 100.0 - 2.0 * (x - 20), 100.0)

    def func_expression_128(x):
        x = float(x)
        if x < 10:
            return 80
        elif x < 15:
            return min_x_max(0.0, 80.0 - 10.0 * (x - 10), 100.0)
        elif x < 20:
            return min_x_max(0.0, 30.0 + 12.0 * (x - 15), 100.0)
        elif x < 25:
            return min_x_max(0.0, 90.0 - 18.0 * (x - 20), 100.0)
        else:
            return min_x_max(0.0, 20.0 * (x - 25), 100.0)

    def func_expression_2001(x):
        return 0

    def func_expression_2002(x):
        return 90*bandwitdh_ratio

    def func_expression_2003(x):
        x = float(x)
        return min_x_max(0.0, 100.0*bandwitdh_ratio - 2.0*bandwitdh_ratio * x, 100.0*bandwitdh_ratio-1)

    def func_expression_2004(x):
        x = float(x)
        return min_x_max(0.0, 2.0*bandwitdh_ratio * x, 100.0*bandwitdh_ratio-1)

    def func_expression_2005(x):
        x = int(x)
        x = x % 20
        x = int(x / 2)
        if x == 0:
            return 0.0*bandwitdh_ratio
        elif x == 1:
            return 90.0*bandwitdh_ratio
        elif x == 2:
            return 10.0*bandwitdh_ratio
        elif x == 3:
            return 80.0*bandwitdh_ratio
        elif x == 4:
            return 20.0*bandwitdh_ratio
        elif x == 5:
            return 70.0*bandwitdh_ratio
        elif x == 6:
            return 30.0*bandwitdh_ratio
        elif x == 7:
            return 60.0*bandwitdh_ratio
        elif x == 8:
            return 40.0*bandwitdh_ratio
        else:
            return 50.0*bandwitdh_ratio

    def func_expression_2006(x):
        x = float(x)
        if x < 10:
            return 50*bandwitdh_ratio
        elif x < 20:
            return min_x_max(0.0*bandwitdh_ratio, 50.0*bandwitdh_ratio - 20.0*bandwitdh_ratio * (x - 10), 100.0*bandwitdh_ratio-1)
        else:
            return min_x_max(0.0*bandwitdh_ratio, 20.0*bandwitdh_ratio * (x - 20), 100.0*bandwitdh_ratio-1)

    def func_expression_2007(x):
        x = float(x)
        if x < 10:
            return 20*bandwitdh_ratio
        elif x < 20:
            return min_x_max(0.0*bandwitdh_ratio, 50.0*bandwitdh_ratio + 2.0*bandwitdh_ratio * (x - 10), 100.0*bandwitdh_ratio-1)
        else:
            return min_x_max(0.0*bandwitdh_ratio, 100.0*bandwitdh_ratio - 2.0*bandwitdh_ratio * (x - 20), 100.0*bandwitdh_ratio-1)

    def func_expression_2008(x):
        x = float(x)
        if x < 10:
            return 80*bandwitdh_ratio
        elif x < 15:
            return min_x_max(0.0*bandwitdh_ratio, 80.0*bandwitdh_ratio - 10.0*bandwitdh_ratio * (x - 10), 100.0*bandwitdh_ratio-1)
        elif x < 20:
            return min_x_max(0.0*bandwitdh_ratio, 30.0*bandwitdh_ratio + 12.0*bandwitdh_ratio * (x - 15), 100.0*bandwitdh_ratio-1)
        elif x < 25:
            return min_x_max(0.0*bandwitdh_ratio, 90.0*bandwitdh_ratio - 18.0*bandwitdh_ratio * (x - 20), 100.0*bandwitdh_ratio-1)
        else:
            return min_x_max(0.0*bandwitdh_ratio, 20.0*bandwitdh_ratio * (x - 25), 100.0*bandwitdh_ratio-1)

    if bound == 0:
        bound = 600

    func_domain = [0.0, bound]  # unit: second
    func_definition = 0.01     # unit: second

    func_1 = Function(func_expression_1, func_domain, func_definition)
    func_2 = Function(func_expression_2, func_domain, func_definition)
    func_3 = Function(func_expression_3, func_domain, func_definition)
    func_4 = Function(func_expression_4, func_domain, func_definition)
    func_5 = Function(func_expression_5, func_domain, func_definition)
    func_6 = Function(func_expression_6, func_domain, func_definition)
    func_7 = Function(func_expression_7, func_domain, func_definition)
    func_8 = Function(func_expression_8, func_domain, func_definition)
    func_9 = Function(func_expression_9, func_domain, func_definition)
    func_10 = Function(func_expression_10, func_domain, func_definition)
    func_11 = Function(func_expression_11, func_domain, func_definition)
    func_12 = Function(func_expression_12, func_domain, func_definition)
    func_13 = Function(func_expression_13, func_domain, func_definition)
    func_14 = Function(func_expression_14, func_domain, func_definition)
    func_15 = Function(func_expression_15, func_domain, func_definition)
    func_16 = Function(func_expression_16, func_domain, func_definition)
    func_17 = Function(func_expression_17, func_domain, func_definition)
    func_18 = Function(func_expression_18, func_domain, func_definition)
    func_19 = Function(func_expression_19, func_domain, func_definition)
    func_20 = Function(func_expression_20, func_domain, func_definition)
    func_21 = Function(func_expression_21, func_domain, func_definition)
    func_22 = Function(func_expression_22, func_domain, func_definition)
    func_23 = Function(func_expression_23, func_domain, func_definition)
    func_24 = Function(func_expression_24, func_domain, func_definition)
    func_25 = Function(func_expression_25, func_domain, func_definition)
    func_26 = Function(func_expression_26, func_domain, func_definition)
    func_27 = Function(func_expression_27, func_domain, func_definition)
    func_28 = Function(func_expression_28, func_domain, func_definition)
    func_29 = Function(func_expression_29, func_domain, func_definition)
    func_30 = Function(func_expression_30, func_domain, func_definition)
    func_31 = Function(func_expression_31, func_domain, func_definition)
    func_32 = Function(func_expression_32, func_domain, func_definition)
    func_33 = Function(func_expression_33, func_domain, func_definition)
    # new train
    func_50 = Function(func_expression_50, func_domain, func_definition)
    func_51 = Function(func_expression_51, func_domain, func_definition)
    func_52 = Function(func_expression_52, func_domain, func_definition)
    func_53 = Function(func_expression_53, func_domain, func_definition)
    func_60 = Function(func_expression_60, func_domain, func_definition)
    func_61 = Function(func_expression_61, func_domain, func_definition)
    func_62 = Function(func_expression_62, func_domain, func_definition)
    func_63 = Function(func_expression_63, func_domain, func_definition)
    func_70 = Function(func_expression_70, func_domain, func_definition)
    func_71 = Function(func_expression_71, func_domain, func_definition)
    func_72 = Function(func_expression_72, func_domain, func_definition)
    func_73 = Function(func_expression_73, func_domain, func_definition)

    # for test
    # for 1000M bw
    func_101 = Function(func_expression_101, func_domain, func_definition)
    func_102 = Function(func_expression_102, func_domain, func_definition)
    func_103 = Function(func_expression_103, func_domain, func_definition)
    func_104 = Function(func_expression_104, func_domain, func_definition)
    func_105 = Function(func_expression_105, func_domain, func_definition)
    func_106 = Function(func_expression_106, func_domain, func_definition)
    func_107 = Function(func_expression_107, func_domain, func_definition)
    func_108 = Function(func_expression_108, func_domain, func_definition)

    # for 500M bw
    func_111 = Function(func_expression_111, func_domain, func_definition)
    func_112 = Function(func_expression_112, func_domain, func_definition)
    func_113 = Function(func_expression_113, func_domain, func_definition)
    func_114 = Function(func_expression_114, func_domain, func_definition)
    func_115 = Function(func_expression_115, func_domain, func_definition)
    func_116 = Function(func_expression_116, func_domain, func_definition)
    func_117 = Function(func_expression_117, func_domain, func_definition)
    func_118 = Function(func_expression_118, func_domain, func_definition)

    # for 100M bw
    func_121 = Function(func_expression_121, func_domain, func_definition)
    func_122 = Function(func_expression_122, func_domain, func_definition)
    func_123 = Function(func_expression_123, func_domain, func_definition)
    func_124 = Function(func_expression_124, func_domain, func_definition)
    func_125 = Function(func_expression_125, func_domain, func_definition)
    func_126 = Function(func_expression_126, func_domain, func_definition)
    func_127 = Function(func_expression_127, func_domain, func_definition)
    func_128 = Function(func_expression_128, func_domain, func_definition)

    # for different bw
    func_1000 = Function(func_expression_1000, func_domain, func_definition)
    func_1001 = Function(func_expression_1001, func_domain, func_definition)
    func_1002 = Function(func_expression_1002, func_domain, func_definition)
    func_1003 = Function(func_expression_1003, func_domain, func_definition)

    func_2001 = Function(func_expression_2001, func_domain, func_definition)
    func_2002 = Function(func_expression_2002, func_domain, func_definition)
    func_2003 = Function(func_expression_2003, func_domain, func_definition)
    func_2004 = Function(func_expression_2004, func_domain, func_definition)
    func_2005 = Function(func_expression_2005, func_domain, func_definition)
    func_2006 = Function(func_expression_2006, func_domain, func_definition)
    func_2007 = Function(func_expression_2007, func_domain, func_definition)
    func_2008 = Function(func_expression_2008, func_domain, func_definition)

    func_list = {1: func_1, 2: func_2, 3: func_3, 4: func_4, 5: func_5, 6: func_6, 7: func_7, 8: func_8, 9: func_9, 10: func_10, 11: func_11,
                 12: func_12, 13: func_13, 14: func_14, 15: func_15, 16: func_16, 17: func_17, 18: func_18, 19: func_19, 20: func_20, 21: func_21, 22: func_22,
                 23: func_23, 24: func_24, 25: func_25, 26: func_26, 27: func_27, 28: func_28, 29: func_29, 30: func_30, 31: func_31, 32: func_32, 33: func_33,
                 50: func_50, 51: func_51, 52: func_52, 53: func_53,
                 60: func_60, 61: func_61, 62: func_62, 63: func_63,
                 70: func_70, 71: func_71, 72: func_72, 73: func_73,
                 101: func_101, 102: func_102, 103: func_103, 104: func_104, 105: func_105, 106: func_106, 107: func_107, 108: func_108,
                 111: func_111, 112: func_112, 113: func_113, 114: func_114, 115: func_115, 116: func_116, 117: func_117, 118: func_118,
                 121: func_121, 122: func_122, 123: func_123, 124: func_124, 125: func_125, 126: func_126, 127: func_127, 128: func_128,
                 1000:func_1000,1001:func_1001,1002:func_1002,1003:func_1003,
                 2001:func_2001,2002:func_2002,2003:func_2003,2004:func_2004,2005:func_2005,2006:func_2006,2007:func_2007,2008:func_2008}

    if shape < 1000:
        return func_list[shape]
    elif shape < 2000:
        bandwitdh_ratio = int(shape/10)-100
        return func_list[shape-bandwitdh_ratio*10]
    elif shape >= 2000:
        bandwitdh_ratio = int(shape/10)-200
        return func_list[shape-bandwitdh_ratio*10]


def generate_traffic(addr, port, dev, shape, bound):
    bind_addr = (addr, port)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    msg = 'x' * 2954
    func = get_func(shape, bound)
    pacing_bin = func.get_pacing_bin()
    bin_len = len(pacing_bin)

    pre_cmd_time = time.time()
    subprocess.Popen('tc qdisc add dev {} root fq maxrate {}'.format(dev, pacing_bin[0]), shell=True)
    cmd_time = time.time() - pre_cmd_time
    sleep_time = func.definition_s - cmd_time

    now = time.time()
    try:
        pid = os.fork()
        if pid == 0:  # child process
            while True:
                sender.sendto(msg, bind_addr)
        else:         # parent process
            for i in xrange(bin_len):
                subprocess.Popen('tc qdisc change dev {} root fq maxrate {}'.format(dev, pacing_bin[i]), shell=True)
                time.sleep(sleep_time)

            commands.getstatusoutput('kill {}'.format(pid))
    except OSError:
        sys.exit(-1)

    exec_time = time.time() - now
    print('exec time: {}'.format(exec_time))
    sender.close()


def main(args):
    # prepare env
    addr = args[1]
    port = int(args[2])
    dev = args[3]
    shape = int(args[4])
    bound = float(args[5])
    commands.getstatusoutput('tc qdisc del dev {} root'.format(dev))
    # start to generate
    generate_traffic(addr, port, dev, shape, bound)


if __name__ == '__main__':
    main(sys.argv)
