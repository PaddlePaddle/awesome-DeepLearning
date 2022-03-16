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
from abc import abstractmethod

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import build_loss
from paddlevideo.utils import get_logger, get_dist_info

logger = get_logger("paddlevideo")


class BaseHead(nn.Layer):
    """Base class for head part.

    All head should subclass it.
    All subclass should overwrite:

    - Methods: ```init_weights```, initializing weights.
    - Methods: ```forward```, forward function.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channels in input feature.
        loss_cfg (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        ls_eps (float): label smoothing epsilon. Default: 0. .

    """
    def __init__(
        self,
        num_classes,
        in_channels,
        loss_cfg=dict(
            name="CrossEntropyLoss"
        ),  #TODO(shipping): only pass a name or standard build cfg format.
        #multi_class=False, NOTE(shipping): not supported now.
        ls_eps=0.):

        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_func = build_loss(loss_cfg)
        #self.multi_class = multi_class NOTE(shipping): not supported now
        self.ls_eps = ls_eps

    @abstractmethod
    def forward(self, x):
        """Define how the head is going to run.
        """
        raise NotImplemented

    def loss(self, scores, labels, valid_mode=False, if_top5=True, **kwargs):
        """Calculate the loss accroding to the model output ```scores```,
           and the target ```labels```.

        Args:
            scores (paddle.Tensor): The output of the model.
            labels (paddle.Tensor): The target output of the model.

        Returns:
            losses (dict): A dict containing field 'loss'(mandatory) and 'top1_acc', 'top5_acc'(optional).

        """
        if len(labels) == 1:  #commonly case
            labels = labels[0]
            losses = dict()
            if self.ls_eps != 0. and not valid_mode:  # label_smooth
                loss = self.label_smooth_loss(scores, labels, **kwargs)
            else:
                loss = self.loss_func(scores, labels, **kwargs)
            if if_top5:
                top1, top5 = self.get_acc(scores, labels, valid_mode)
                losses['top1'] = top1
                losses['top5'] = top5
                losses['loss'] = loss
            else:
                top1 = self.get_acc(scores, labels, valid_mode, if_top5)
                losses['top1'] = top1
                losses['loss'] = loss
            return losses
        # MRI目前二分类无top5
        elif len(labels) == 3:  # mix_up
            labels_a, labels_b, lam = labels
            lam = lam[0]  # get lam value
            losses = dict()
            if self.ls_eps != 0:
                loss_a = self.label_smooth_loss(scores, labels_a, **kwargs)
                loss_b = self.label_smooth_loss(scores, labels_b, **kwargs)
            else:
                loss_a = self.loss_func(scores, labels_a, **kwargs)
                loss_b = self.loss_func(scores, labels_b, **kwargs)
            loss = lam * loss_a + (1 - lam) * loss_b

            if if_top5:
                top1a, top5a = self.get_acc(scores, labels_a, valid_mode)
                top1b, top5b = self.get_acc(scores, labels_b, valid_mode)
                top1 = lam * top1a + (1 - lam) * top1b
                top5 = lam * top5a + (1 - lam) * top5b
                losses['top1'] = top1
                losses['top5'] = top5
                losses['loss'] = loss

            else:
                top1a = self.get_acc(scores, labels_a, valid_mode, if_top5)
                top1b = self.get_acc(scores, labels_b, valid_mode, if_top5)
                top1 = lam * top1a + (1 - lam) * top1b
                losses['top1'] = top1
                losses['loss'] = loss

            return losses
        else:
            raise NotImplemented

    def label_smooth_loss(self, scores, labels, **kwargs):
        """
        Args:
            scores (paddle.Tensor): [N, num_classes]
            labels (paddle.Tensor): [N, ]
        Returns:
            paddle.Tensor: [1,]
        """
        if paddle.fluid.core.is_compiled_with_npu():
            """
            Designed for the lack of temporary operators of NPU,
            main idea is to split smooth loss into uniform distribution loss
            and hard label calculation
            """
            hard_loss = (1.0 - self.ls_eps) * F.cross_entropy(scores, labels)
            uniform_loss = (self.ls_eps / self.num_classes) * (
                -F.log_softmax(scores, -1).sum(-1).mean(0))
            loss = hard_loss + uniform_loss
        else:
            labels = F.one_hot(labels, self.num_classes)
            labels = F.label_smooth(labels, epsilon=self.ls_eps)
            labels = paddle.squeeze(labels, axis=1)
            loss = self.loss_func(scores, labels, soft_label=True, **kwargs)
        return loss

    def get_acc(self, scores, labels, valid_mode, if_top5=True):
        if if_top5:
            top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
            top5 = paddle.metric.accuracy(input=scores, label=labels, k=5)
            _, world_size = get_dist_info()
            #NOTE(shipping): deal with multi cards validate
            if world_size > 1 and valid_mode:  #reduce sum when valid
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / world_size
                top5 = paddle.distributed.all_reduce(
                    top5, op=paddle.distributed.ReduceOp.SUM) / world_size

            return top1, top5
        else:
            top1 = paddle.metric.accuracy(input=scores, label=labels, k=1)
            _, world_size = get_dist_info()
            #NOTE(shipping): deal with multi cards validate
            if world_size > 1 and valid_mode:  #reduce sum when valid
                top1 = paddle.distributed.all_reduce(
                    top1, op=paddle.distributed.ReduceOp.SUM) / world_size

            return top1
