# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import paddle
from . import custom_lr


def build_lr(cfg, num_iters):
    """
    Build a learning rate scheduler accroding to ```OPTIMIZER``` configuration, and it always pass into the optimizer.
    In configuration:
    learning_rate:
        name: 'PiecewiseDecay'
        boundaries: [20, 60]
        values: [0.00025, 0.000025, 0.0000025]


    Returns:
        A paddle.optimizer.lr instance.
    """

    cfg_copy = cfg.copy()

    #when learning_rate is LRScheduler
    if cfg_copy.get('learning_rate') and isinstance(cfg_copy['learning_rate'],
                                                    dict):
        cfg_copy['learning_rate'] = build_lr(
            cfg_copy['learning_rate'],
            num_iters)  #not support only inner iter_step

    lr_name = cfg_copy.pop('name')
    if cfg_copy.get('iter_step'):
        cfg_copy['num_iters'] = num_iters
        cfg_copy.pop('iter_step')

    return getattr(custom_lr, lr_name)(**cfg_copy)
