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

import paddle
import itertools

from paddlevideo.utils import get_logger
logger = get_logger("paddlevideo")
"""
Implement precise bn, which is useful for improving accuracy.
"""


@paddle.no_grad()  # speed up and save CUDA memory
def do_preciseBN(model, data_loader, parallel, num_iters=200):
    """
    Recompute and update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every iteration, so
    the running average can not precisely reflect the actual stats of the
    current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    This is useful to improve validation accuracy.
    Args:
        model: the model whose bn stats will be recomputed
        data_loader: an iterator. Produce data as input to the model
        num_iters: number of iterations to compute the stats.
    Return:
        the model with precise mean and variance in bn layers.
    """
    bn_layers_list = [
        m for m in model.sublayers()
        if any((isinstance(m, bn_type)
                for bn_type in (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D,
                                paddle.nn.BatchNorm3D))) and m.training
    ]
    if len(bn_layers_list) == 0:
        return

    # moving_mean=moving_mean*momentum+batch_mean*(1.−momentum)
    # we set momentum=0. to get the true mean and variance during forward
    momentum_actual = [bn._momentum for bn in bn_layers_list]
    for bn in bn_layers_list:
        bn._momentum = 0.

    running_mean = [paddle.zeros_like(bn._mean)
                    for bn in bn_layers_list]  #pre-ignore
    running_var = [paddle.zeros_like(bn._variance) for bn in bn_layers_list]

    ind = -1
    for ind, data in enumerate(itertools.islice(data_loader, num_iters)):
        logger.info("doing precise BN {} / {}...".format(ind + 1, num_iters))
        if parallel:
            model._layers.train_step(data)
        else:
            model.train_step(data)

        for i, bn in enumerate(bn_layers_list):
            # Accumulates the bn stats.
            running_mean[i] += (bn._mean - running_mean[i]) / (ind + 1)
            running_var[i] += (bn._variance - running_var[i]) / (ind + 1)

    assert ind == num_iters - 1, (
        "update_bn_stats is meant to run for {} iterations, but the dataloader stops at {} iterations."
        .format(num_iters, ind))

    # Sets the precise bn stats.
    for i, bn in enumerate(bn_layers_list):
        bn._mean.set_value(running_mean[i])
        bn._variance.set_value(running_var[i])
        bn._momentum = momentum_actual[i]
