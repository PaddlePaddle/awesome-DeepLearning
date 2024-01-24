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

import paddle
import paddle.nn.functional as F
from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class TransNetV2Loss(BaseWeightedLoss):
    """Loss for TransNetV2 model
    """
    def __init__(self, transition_weight=5.0, many_hot_loss_weight=0.1):
        self.transition_weight = transition_weight
        self.many_hot_loss_weight = many_hot_loss_weight
        super().__init__()

    def _forward(self, one_hot_pred, one_hot_gt,
                many_hot_pred=None, many_hot_gt=None, reg_losses=None):
        assert transition_weight != 1

        one_hot_pred = one_hot_pred[:, :, 0]

        one_hot_gt = one_hot_gt.astype('float32')
        one_hot_loss = F.binary_cross_entropy_with_logits(logit=one_hot_pred, label=one_hot_gt, reduction='none')

        one_hot_loss *= 1 + one_hot_gt * (transition_weight - 1)

        one_hot_loss = paddle.mean(one_hot_loss)

        many_hot_loss = 0.
        if many_hot_loss_weight != 0. and many_hot_pred is not None:
            many_hot_loss = many_hot_loss_weight * paddle.mean(
                F.binary_cross_entropy_with_logits(logit=many_hot_pred[:, :, 0],
                                                   label=many_hot_gt.astype('float32'), reduction='none'))

        total_loss = one_hot_loss + many_hot_loss

        if reg_losses is not None:
            for name, value in reg_losses.items():
                if value is not None:
                    total_loss += value

        return total_loss