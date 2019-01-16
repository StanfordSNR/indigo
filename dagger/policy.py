# Copyright 2018 Francis Y. Yan
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


import collections
import datetime
import sys

import context  # noqa # pylint: disable=unused-import
from helpers.utils import Config, format_actions, timestamp_ms, update_ewma
from message import Message


class Policy(object):
    min_cwnd = 2.0
    max_cwnd = 25000.0

    max_rtt = 300.0  # ms
    max_delay = max_rtt
    max_send_rate = 1000.0  # Mbps
    max_delivery_rate = max_send_rate

    min_step_len = 10  # ms
    steps_per_episode = 1000  # number of steps in each episode (in training)

    # state = [rtt_norm, delay_norm, send_rate_norm, delivery_rate_norm,
    #          loss_rate_norm, cwnd_norm]
    state_dim = Config.state_dim * Config.state_his
    label_dim = 3  # len([cwnd, expert_cwnd, expert_action])
    action_list = ["/2.0", "/1.05", "+0.0", "*1.05", "*2.0"]
    action_cnt = len(action_list)
    action_mapping = format_actions(action_list)
    action_frequency = Config.action_frequency

    delay_ack = True
    delay_ack_count = 2

    def __init__(self, train):
        # public:
        self.cwnd = 10.0
        self.bytes_sent = 0
        self.ack_recv_ts = 0
        self.bytes_acked = 0

        # sender should stop or not
        self.stop_sender = False

        # pacing or not
        self.pacing = True

    # private:
        self.train = train
        self.sample_action = None

        # step timer and counting
        self.step_start_ts = None
        self.step_num = 0
        self.start_phase_cnt = 0
        self.start_phase_max = 4  # max number of steps in start phase

        # state related (persistent across steps)
        self.min_rtt = sys.maxint
        self.max_rtt = 0
        self.min_delay_ewma = float('inf')
        self.max_delay_ewma = 0.0
        self.min_send_rate_ewma = float('inf')
        self.max_send_rate_ewma = 0.0
        self.min_delivery_rate_ewma = float('inf')
        self.max_delivery_rate_ewma = 0.0

        # variables to calculate loss rate (persistent across steps)
        self.prev_bytes_acked = 0
        self.prev_bytes_sent_in_ack = 0
        self.bytes_sent_in_ack = 0

        # state related (reset at each step)
        self.rtt_ewma = None
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None
        self.history_state = collections.deque()

        # pacing related
        self.borrowed_pkt = 0.0
        self.interval = 500.0  # us
        self.pre_sent_ts = None

        # measurement client n in test mode
        self.perf_client = None

        # rtt re-update
        self.rtt_reupdate_interval = 50  # RTT_ewma
        self.rtt_reupdate_start_ts = None

# private
    def __update_state(self, ack):
        # update RTT and queuing delay (in ms)
        rtt = max(1, self.ack_recv_ts - ack.send_ts)
        self.min_rtt = min(self.min_rtt, rtt)
        self.max_rtt = max(self.max_rtt, rtt)
        self.rtt_ewma = update_ewma(self.rtt_ewma, rtt)

        queuing_delay = rtt - self.min_rtt
        self.delay_ewma = update_ewma(self.delay_ewma, queuing_delay)

        self.min_delay_ewma = min(self.min_delay_ewma, self.delay_ewma)
        self.max_delay_ewma = max(self.max_delay_ewma, self.delay_ewma)

        # update sending rate (in Mbps)
        send_rate = 0.008 * (self.bytes_sent - ack.bytes_sent) / rtt
        self.send_rate_ewma = update_ewma(self.send_rate_ewma, send_rate)

        self.min_send_rate_ewma = min(self.min_send_rate_ewma,
                                      self.send_rate_ewma)
        self.max_send_rate_ewma = max(self.max_send_rate_ewma,
                                      self.send_rate_ewma)

        # update delivery rate (in Mbps)
        duration = max(1, self.ack_recv_ts - ack.ack_recv_ts)
        delivery_rate = 0.008 * (self.bytes_acked - ack.bytes_acked) / duration
        self.delivery_rate_ewma = update_ewma(self.delivery_rate_ewma,
                                              delivery_rate)

        self.min_delivery_rate_ewma = min(self.min_delivery_rate_ewma,
                                          self.delivery_rate_ewma)
        self.max_delivery_rate_ewma = max(self.max_delivery_rate_ewma,
                                          self.delivery_rate_ewma)

        # record the sent bytes in the current ACK
        self.bytes_sent_in_ack = ack.bytes_sent

    # calculate loss rate at the end of each step
    # step_acked = acked bytes during this step
    # step_sent = sent bytes recorded in ACKs received at this step
    # loss = 1 - step_acked / step_sent
    def __cal_loss_rate(self):
        step_sent = self.bytes_sent_in_ack - self.prev_bytes_sent_in_ack
        if step_sent == 0:  # prevent divide-by-0 error
            return 0

        step_acked = self.bytes_acked - self.prev_bytes_acked
        loss_rate = 1.0 - float(step_acked) / step_sent

        self.prev_bytes_acked = self.bytes_acked
        self.prev_bytes_sent_in_ack = self.bytes_sent_in_ack

        # in case packet reordering occurred
        return min(max(0.0, loss_rate), 1.0)

    def __take_action(self, action):
        if action < 0 or action >= Policy.action_cnt:
            sys.exit('invalid action')

        op, val = Policy.action_mapping[action]
        self.cwnd = op(self.cwnd, val)
        self.cwnd = max(Policy.min_cwnd, min(Policy.max_cwnd, self.cwnd))

    # reset some stats at each step
    def __reset_step(self):
        self.rtt_ewma = None
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

    def __episode_ended(self):
        self.stop_sender = True

    def __step_ended(self):
        if self.stop_sender:
            return
        # normalization
        rtt_norm = self.rtt_ewma / Policy.max_rtt
        delay_norm = self.delay_ewma / Policy.max_rtt
        max_rtt_norm = self.max_rtt / Policy.max_rtt
        min_rtt_norm = self.min_rtt / Policy.max_rtt
        send_rate_norm = self.send_rate_ewma / Policy.max_send_rate
        delivery_rate_norm = self.delivery_rate_ewma / Policy.max_delivery_rate
        cwnd_norm = self.cwnd / Policy.max_cwnd
        loss_rate_norm = self.__cal_loss_rate()  # loss is already in [0, 1]

        # state -> action
        state_array = [[rtt_norm, delay_norm, send_rate_norm, delivery_rate_norm,
                        loss_rate_norm, cwnd_norm],
                       [rtt_norm, delay_norm, max_rtt_norm, min_rtt_norm, send_rate_norm, delivery_rate_norm,
                        loss_rate_norm, cwnd_norm]]
        state = state_array[Config.state_idx]

        # print state,self.min_rtt,self.max_rtt
        if len(self.history_state) == 0:
            for _ in xrange(Config.state_his):
                self.history_state.append(state)
        elif len(self.history_state) == Config.state_his:
            self.history_state.popleft()
            self.history_state.append(state)
        h_state = []
        for s in self.history_state:
            h_state = h_state + s

        if self.sample_action is None:
            sys.exit('sample_action on policy has not been set')
        action = self.sample_action(h_state)

        self.__take_action(action)

        # reset at the end of each step
        self.__reset_step()

        # step counting
        if self.train:
            self.step_num += 1
            if self.step_num >= Policy.steps_per_episode:
                self.__episode_ended()

    def __is_step_ended(self, duration):
        if self.train:
            return duration >= Policy.min_step_len
        else:
            # cwnd is updated every RTT in start phase, and min_rtt/action_frequency afterwards
            if self.start_phase_cnt < self.start_phase_max:
                threshold = max(self.min_rtt, Policy.min_step_len)
                self.start_phase_cnt += 1
            else:
                threshold = max(self.min_rtt / Policy.action_frequency, Policy.min_step_len)

            return duration >= threshold

    def __re_update_rtt(self, curr_ts):
        if self.rtt_reupdate_start_ts is None:
            self.rtt_reupdate_start_ts = curr_ts
            return

        if curr_ts - self.rtt_reupdate_start_ts > self.rtt_reupdate_interval * self.min_rtt:
            print curr_ts - self.rtt_reupdate_start_ts, 'Min RTT:', self.min_rtt, 'Max RTT:', self.max_rtt
            self.rtt_reupdate_start_ts = curr_ts
            self.min_rtt = sys.maxint
            self.max_rtt = 0

# public:
    def ack_received(self, ack):
        self.ack_recv_ts = timestamp_ms()
        if Policy.delay_ack:
            self.bytes_acked += Message.total_size * Policy.delay_ack_count
        else:
            self.bytes_acked += Message.total_size

        self.__update_state(ack)

        curr_ts = timestamp_ms()
        if self.step_start_ts is None:
            self.step_start_ts = curr_ts

        # check if the current step has ended
        if self.__is_step_ended(curr_ts - self.step_start_ts):
            # if self.perf_client:
            #     self.perf_client.collect_perf_data(self)

            self.step_start_ts = curr_ts
            self.__step_ended()

        # self.__re_update_rtt(curr_ts)

    def data_sent(self):
        self.bytes_sent += Message.total_size

    def timeout_ms(self):
        return 10

    def set_sample_action(self, sample_action):
        self.sample_action = sample_action

    def set_perf_client(self, perf_client):
        self.perf_client = perf_client

    def pacing_pkt_number(self, max_in_cwnd):
        # pacing control
        max_in_cwnd = int(max_in_cwnd)

        if not self.pacing or self.min_rtt == sys.maxint:
            return max_in_cwnd

        if self.pre_sent_ts is None:
            self.pre_sent_ts = datetime.datetime.now()

        now = datetime.datetime.now()
        duration = (now - self.pre_sent_ts).microseconds
        if duration >= self.interval:
            pacing_rate = self.cwnd / self.min_rtt  # pkt/ms
            n = duration / 1000.0 * pacing_rate

            self.borrowed_pkt += n - int(n)
            n = int(n)
            if self.borrowed_pkt >= 1.0:
                n += int(self.borrowed_pkt)
                self.borrowed_pkt = self.borrowed_pkt - int(self.borrowed_pkt)
            ret_num = min(max_in_cwnd, n)

            self.pre_sent_ts = now
        else:
            ret_num = 0

        return ret_num
