# Copyright 2018 Wei Wang, Yiyang Shao (Huawei Technologies)
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

from helpers.utils import Config
from message import Message


class Oracle():
    def __init__(self, net_config):
        self.LINK_CAPACITY, self.MIN_RTT, self.MAX_QUEUE_SIZE = net_config

    def get_oracle(self, net_info):
        pass


class StandardBDP(Oracle):
    """ best_cwnd = available_bw * min_rtt """

    def get_oracle(self, net_info):
        available_bw_bn, throughput_tg, queueing_factor_bn, random_loss_bn, congestion_loss_bn, queue_size_bn = net_info

        best_send_rate = self.LINK_CAPACITY - throughput_tg  # Mbps
        if best_send_rate < 0:
            best_send_rate = 0

        # Mbps * ms = 10^6 b/s * 10^(-3) s = b
        std_bdp = 1000 * best_send_rate * self.MIN_RTT / 8.0 / Message.total_size

        return std_bdp


class AggressiveBDP(Oracle):
    """ best_cwnd = available_bw * min_rtt + q * left_queue_size, q is the aggressive factor and 0 <= q <= 1 """

    def get_oracle(self, net_info):
        available_bw_bn, throughput_tg, queueing_factor_bn, random_loss_bn, congestion_loss_bn, queue_size_bn = net_info
        q = Config.fri

        best_send_rate = self.LINK_CAPACITY - throughput_tg  # Mbps
        if best_send_rate < 0:
            best_send_rate = 0

        # Mbps * ms = 10^6 b/s * 10^(-3) s = b
        std_bdp = 1000 * best_send_rate * self.MIN_RTT / 8.0 / Message.total_size

        aggressive_bdp = std_bdp + (self.MAX_QUEUE_SIZE * q - self.MAX_QUEUE_SIZE * queueing_factor_bn)

        return aggressive_bdp
