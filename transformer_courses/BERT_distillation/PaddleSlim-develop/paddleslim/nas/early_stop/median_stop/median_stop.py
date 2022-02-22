#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from multiprocessing.managers import BaseManager
from ..early_stop import EarlyStopBase
from ....common.log_helper import get_logger

PublicAuthKey = u'AbcXyz3'

__all__ = ['MedianStop']

_logger = get_logger(__name__, level=logging.INFO)

completed_history = dict()


def return_completed_history():
    return completed_history


class MedianStop(EarlyStopBase):
    """
    Median Stop, reference:
    Args:
        strategy<class instance>: the stategy of search.
        start_epoch<int>: which step to start early stop algorithm.
        mode<str>: bigger is better or smaller is better, chooice in ['maxmize', 'minimize']. Default: maxmize.
    """

    def __init__(self, strategy, start_epoch, mode='maxmize'):
        self._start_epoch = start_epoch
        self._running_history = dict()
        self._strategy = strategy
        self._mode = mode
        self._is_server = self._strategy._is_server
        self._manager = self._start_manager()
        assert self._mode in [
            'maxmize', 'minimize'
        ], 'mode of MedianStop must be \'maxmize\' or \'minimize\', but received mode is {}'.format(
            self._mode)

    def _start_manager(self):
        self._server_ip = self._strategy._server_ip
        self._server_port = self._strategy._server_port + 1

        if self._is_server:
            BaseManager.register(
                'get_completed_history', callable=return_completed_history)
            base_manager = BaseManager(
                address=(self._server_ip, self._server_port),
                authkey=PublicAuthKey.encode())

            base_manager.start()
        else:
            BaseManager.register('get_completed_history')
            base_manager = BaseManager(
                address=(self._server_ip, self._server_port),
                authkey=PublicAuthKey.encode())
            base_manager.connect()
        return base_manager

    def _update_data(self, exp_name, result):
        if exp_name not in self._running_history.keys():
            self._running_history[exp_name] = []
        self._running_history[exp_name].append(result)

    def _convert_running2completed(self, exp_name, status):
        """
        Convert experiment record from running to complete.

        Args:
           exp_name<str>: the name of experiment.
           status<str>: the status of this experiment.
        """
        _logger.debug('the status of this experiment is {}'.format(status))
        completed_avg_history = dict()
        if exp_name in self._running_history:
            if status == "GOOD":
                count = 0
                history_sum = 0
                result = []
                for res in self._running_history[exp_name]:
                    count += 1
                    history_sum += res
                    result.append(history_sum / count)
                completed_avg_history[exp_name] = result
            self._running_history.pop(exp_name)

        if len(completed_avg_history) > 0:
            while True:
                try:
                    new_dict = self._manager.get_completed_history()
                    new_dict.update(completed_avg_history)
                    break
                except Exception as err:
                    _logger.error("update data error: {}".format(err))

    def get_status(self, step, result, epochs):
        """ 
        Get current experiment status
        
        Args:
            step: step in this client.
            result: the result of this epoch.
            epochs: whole epochs.

        Return:
            the status of this experiment.
        """
        exp_name = self._strategy._client_name + str(step)
        self._update_data(exp_name, result)

        _logger.debug("running history after update data: {}".format(
            self._running_history))

        curr_step = len(self._running_history[exp_name])
        status = "GOOD"
        if curr_step < self._start_epoch:
            return status

        res_same_step = []

        def list2dict(lists):
            res_dict = dict()
            for l in lists:
                tmp_dict = dict()
                tmp_dict[l[0]] = l[1]
                res_dict.update(tmp_dict)
            return res_dict

        while True:
            try:
                completed_avg_history = self._manager.get_completed_history()
                break
            except Exception as err:
                _logger.error("get status error: {}".format(err))

        if len(completed_avg_history.keys()) == 0:
            for exp in self._running_history.keys():
                if curr_step <= len(self._running_history[exp]):
                    res_same_step.append(self._running_history[exp][curr_step -
                                                                    1])
        else:
            completed_avg_history_dict = list2dict(completed_avg_history.items(
            ))

            for exp in completed_avg_history.keys():
                if curr_step <= len(completed_avg_history_dict[exp]):
                    res_same_step.append(completed_avg_history_dict[exp][
                        curr_step - 1])

        _logger.debug("result of same step in other experiment: {}".format(
            res_same_step))
        if res_same_step:
            res_same_step.sort()

            if self._mode == 'maxmize' and result < res_same_step[(
                    len(res_same_step) - 1) // 2]:
                status = "BAD"

            if self._mode == 'minimize' and result > res_same_step[len(
                    res_same_step) // 2]:
                status = "BAD"

        if curr_step == epochs:
            self._convert_running2completed(exp_name, status)

        return status

    def __del__(self):
        if self._is_server:
            self._manager.shutdown()
