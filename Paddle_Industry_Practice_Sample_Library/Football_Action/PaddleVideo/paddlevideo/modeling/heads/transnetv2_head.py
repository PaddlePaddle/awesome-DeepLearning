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

from .base import BaseHead
from ..registry import HEADS
from ..losses import TransNetV2Loss
from ...metrics.transnetv2_metric import create_scene_based_summaries

@HEADS.register()
class TransNetV2Head(BaseHead):
    """TransNetV2 Head.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name="TransNetV2Loss")
                 ):
        super().__init__(num_classes,
                         in_channels,
                         loss_cfg)

    def loss(self, one_hot_pred, one_hot_gt,
                many_hot_pred=None, many_hot_gt=None, reg_losses=None):
        losses = dict()
        loss = self.loss_func(scores, labels, **kwargs)

        f1 = self.get_score(one_hot_pred, one_hot_gt)
        losses['f1'] = f1
        losses['loss'] = loss
        return losses

    def get_score(self, one_hot_pred, one_hot_gt):
        f1 = create_scene_based_summaries(one_hot_pred, one_hot_gt)
        return f1
