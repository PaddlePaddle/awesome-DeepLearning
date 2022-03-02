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

from paddle.nn import Linear

from ..registry import HEADS
from ..weight_init import trunc_normal_, weight_init_
from .base import BaseHead
from paddle import ParamAttr
from paddle.regularizer import L2Decay


@HEADS.register()
class ppTimeSformerHead(BaseHead):
    """TimeSformerHead Head.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        std(float): Std(Scale) value in normal initilizar. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to initialize.

    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 std=0.02,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cfg, **kwargs)
        self.std = std
        self.fc = Linear(self.in_channels,
                         self.num_classes,
                         bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

    def init_weights(self):
        """Initiate the FC layer parameters"""

        weight_init_(self.fc,
                     'TruncatedNormal',
                     'fc_0.w_0',
                     'fc_0.b_0',
                     mean=0.0,
                     std=self.std)
        # NOTE: Temporarily use trunc_normal_ instead of TruncatedNormal
        trunc_normal_(self.fc.weight, std=self.std)

    def forward(self, x):
        """Define how the head is going to run.
        Args:
            x (paddle.Tensor): The input data.
        Returns:
            score: (paddle.Tensor) The classification scores for input samples.
        """
        # XXX: check dropout location!
        # x.shape = [N, embed_dim]

        score = self.fc(x)
        # [N, num_class]
        # x = F.softmax(x)  # NOTE remove
        return score
