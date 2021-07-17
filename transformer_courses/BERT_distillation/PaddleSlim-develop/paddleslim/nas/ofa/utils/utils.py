#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddle
from ....common import get_logger


def get_paddle_version():
    import paddle
    pd_ver = 185
    if hasattr(paddle, 'nn'):
        if hasattr(paddle.nn, 'Conv1D'):  ### judge 2.0 alpha
            pd_ver = 200

    return pd_ver


pd_ver = get_paddle_version()
if pd_ver == 185:
    Layer = paddle.fluid.dygraph.Layer
else:
    Layer = paddle.nn.Layer

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['set_state_dict']


def set_state_dict(model, state_dict):
    """
    Set state dict from origin model to supernet model.

    Args:
        model(paddle.nn.Layer): model after convert to supernet.
        state_dict(dict): dict with the type of {name: param} in origin model.
    """
    assert isinstance(model, Layer)
    assert isinstance(state_dict, dict)
    for name, param in model.state_dict().items():
        tmp_n = name.split('.')[:-2] + [name.split('.')[-1]]
        tmp_n = '.'.join(tmp_n)
        if name in state_dict:
            param.set_value(state_dict[name])
        elif tmp_n in state_dict:
            param.set_value(state_dict[tmp_n])
        else:
            _logger.info('{} is not in state_dict'.format(tmp_n))


def remove_model_fn(model, state_dict):
    new_dict = {}
    keys = []
    for name, param in model.state_dict().items():
        keys.append(name)
    for name, param in state_dict.items():
        if len(name.split('.')) <= 2:
            new_dict[name] = param
            continue
        if name.split('.')[-2] == 'fn':
            tmp_n = name.split('.')[:-2] + [name.split('.')[-1]]
            tmp_n = '.'.join(tmp_n)
        if name in keys:
            new_dict[name] = param
        elif tmp_n in keys:
            new_dict[tmp_n] = param
        else:
            _logger.debug('{} is not in state_dict'.format(tmp_n))
    return new_dict


def compute_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    sub_center = sub_kernel_size // 2
    start = center - sub_center
    if sub_kernel_size % 2 == 0:
        end = center + sub_center
    else:
        end = center + sub_center + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    assert isinstance(kernel_size, int)
    assert kernel_size % 2 > 0, "kernel size must be odd number"
    return kernel_size // 2


def convert_to_list(value, n):
    return [value, ] * n


def search_idx(num, sorted_nestlist):
    max_num = -1
    max_idx = -1
    for idx in range(len(sorted_nestlist)):
        task_ = sorted_nestlist[idx]
        max_num = task_[-1]
        max_idx = len(task_) - 1
        for phase_idx in range(len(task_)):
            if num <= task_[phase_idx]:
                return idx, phase_idx
    assert num > max_num
    return len(sorted_nestlist) - 1, max_idx
