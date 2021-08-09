# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
from ...core import Registry

__all__ = [
    "RLCONTROLLER", "action_mapping", "add_grad", "compute_grad",
    "ConnectMessage"
]

RLCONTROLLER = Registry('RLController')


class ConnectMessage:
    INIT = 'INIT'
    INIT_DONE = 'INIT_DONE'
    GET_WEIGHT = 'GET_WEIGHT'
    UPDATE_WEIGHT = 'UPDATE_WEIGHT'
    OK = 'OK'
    WAIT = 'WAIT'
    WAIT_PARAMS = 'WAIT_PARAMS'
    EXIT = 'EXIT'
    TIMEOUT = 10


def action_mapping(actions, range_table):
    actions = (actions - (-1.0)) * (range_table / np.asarray(2.0))
    return actions.astype('int64')


def add_grad(dict1, dict2):
    dict3 = dict()
    for key, value in dict1.items():
        dict3[key] = dict1[key] + dict2[key]
    return dict3


def compute_grad(dict1, dict2):
    dict3 = dict()
    for key, value in dict1.items():
        dict3[key] = dict1[key] - dict2[key]
    return dict3
