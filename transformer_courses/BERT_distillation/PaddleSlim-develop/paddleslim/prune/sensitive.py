# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import logging
import pickle
import numpy as np
import paddle
from ..core import GraphWrapper
from ..common import get_logger
from ..analysis import flops
from ..prune import Pruner

_logger = get_logger(__name__, level=logging.INFO)

__all__ = [
    "sensitivity", "load_sensitivities", "merge_sensitive", "get_ratios_by_loss"
]


def sensitivity(program,
                place,
                param_names,
                eval_func,
                sensitivities_file=None,
                pruned_ratios=None,
                eval_args=None,
                criterion='l1_norm'):
    """Compute the sensitivities of convolutions in a model. The sensitivity of a convolution is the losses of accuracy on test dataset in differenct pruned ratios. The sensitivities can be used to get a group of best ratios with some condition.
    This function return a dict storing sensitivities as below:

    .. code-block:: python

           {"weight_0":
               {0.1: 0.22,
                0.2: 0.33
               },
             "weight_1":
               {0.1: 0.21,
                0.2: 0.4
               }
           }

    ``weight_0`` is parameter name of convolution. ``sensitivities['weight_0']`` is a dict in which key is pruned ratio and value is the percent of losses.


    Args:
        program(paddle.static.Program): The program to be analysised.
        place(paddle.CPUPlace | paddle.CUDAPlace): The device place of filter parameters. 
        param_names(list): The parameter names of convolutions to be analysised. 
        eval_func(function): The callback function used to evaluate the model. It should accept a instance of `paddle.static.Program` as argument and return a score on test dataset.
        sensitivities_file(str): The file to save the sensitivities. It will append the latest computed sensitivities into the file. And the sensitivities in the file would not be computed again. This file can be loaded by `pickle` library.
        pruned_ratios(list): The ratios to be pruned. default: ``[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]``.

    Returns: 
        dict: A dict storing sensitivities.
    """
    scope = paddle.static.global_scope()
    graph = GraphWrapper(program)
    sensitivities = load_sensitivities(sensitivities_file)

    if pruned_ratios is None:
        pruned_ratios = np.arange(0.1, 1, step=0.1)

    for name in param_names:
        if name not in sensitivities:
            sensitivities[name] = {}
    baseline = None
    for name in sensitivities:
        for ratio in pruned_ratios:
            if ratio in sensitivities[name]:
                _logger.debug('{}, {} has computed.'.format(name, ratio))
                continue
            if baseline is None:
                if eval_args is None:
                    baseline = eval_func(graph.program)
                else:
                    baseline = eval_func(eval_args)

            pruner = Pruner(criterion=criterion)
            _logger.info("sensitive - param: {}; ratios: {}".format(name,
                                                                    ratio))
            pruned_program, param_backup, _ = pruner.prune(
                program=graph.program,
                scope=scope,
                params=[name],
                ratios=[ratio],
                place=place,
                lazy=False,
                only_graph=False,
                param_backup=True)
            if eval_args is None:
                pruned_metric = eval_func(pruned_program)
            else:
                pruned_metric = eval_func(eval_args)
            loss = (baseline - pruned_metric) / baseline
            _logger.info("pruned param: {}; {}; loss={}".format(name, ratio,
                                                                loss))
            sensitivities[name][ratio] = loss

            _save_sensitivities(sensitivities, sensitivities_file)

            # restore pruned parameters
            for param_name in param_backup.keys():
                param_t = scope.find_var(param_name).get_tensor()
                param_t.set(param_backup[param_name], place)
    return sensitivities


def merge_sensitive(sensitivities):
    """Merge sensitivities.

    Args:
      sensitivities(list<dict> | list<str>): The sensitivities to be merged. It cann be a list of sensitivities files or dict.

    Returns:
      dict: A dict stroring sensitivities.
    """
    assert len(sensitivities) > 0
    if not isinstance(sensitivities[0], dict):
        sensitivities = [load_sensitivities(sen) for sen in sensitivities]

    new_sensitivities = {}
    for sen in sensitivities:
        for param, losses in sen.items():
            if param not in new_sensitivities:
                new_sensitivities[param] = {}
            for percent, loss in losses.items():
                new_sensitivities[param][percent] = loss
    return new_sensitivities


def load_sensitivities(sensitivities_file):
    """Load sensitivities from file.

    Args:
       sensitivities_file(str):  The file storing sensitivities.

    Returns:
       dict: A dict stroring sensitivities.
    """
    sensitivities = {}
    if sensitivities_file and os.path.exists(sensitivities_file):
        with open(sensitivities_file, 'rb') as f:
            if sys.version_info < (3, 0):
                sensitivities = pickle.load(f)
            else:
                sensitivities = pickle.load(f, encoding='bytes')
    return sensitivities


def _save_sensitivities(sensitivities, sensitivities_file):
    """Save sensitivities into file.
    
    Args:
        sensitivities(dict): The sensitivities to be saved.
        sensitivities_file(str): The file to saved sensitivities.
    """
    with open(sensitivities_file, 'wb') as f:
        pickle.dump(sensitivities, f)


def get_ratios_by_loss(sensitivities, loss):
    """
    Get the max ratio of each parameter. The loss of accuracy must be less than given `loss`
    when the single parameter was pruned by the max ratio. 
    
    Args:
      
      sensitivities(dict): The sensitivities used to generate a group of pruning ratios. The key of dict
                           is name of parameters to be pruned. The value of dict is a list of tuple with
                           format `(pruned_ratio, accuracy_loss)`.
      loss(float): The threshold of accuracy loss.

    Returns:

      dict: A group of ratios. The key of dict is name of parameters while the value is the ratio to be pruned.
    """
    ratios = {}
    for param, losses in sensitivities.items():
        losses = losses.items()
        losses = list(losses)
        losses.sort()
        for i in range(len(losses))[::-1]:
            if losses[i][1] <= loss:
                if i == (len(losses) - 1):
                    ratios[param] = losses[i][0]
                else:
                    r0, l0 = losses[i]
                    r1, l1 = losses[i + 1]
                    d0 = loss - l0
                    d1 = l1 - loss

                    ratio = r0 + (loss - l0) * (r1 - r0) / (l1 - l0)
                    ratios[param] = ratio
                    if ratio > 1:
                        _logger.info(losses, ratio, (r1 - r0) / (l1 - l0), i)

                break
    return ratios
