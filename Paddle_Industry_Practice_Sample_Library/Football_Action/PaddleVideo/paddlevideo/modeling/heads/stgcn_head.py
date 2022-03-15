# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


@HEADS.register()
class STGCNHead(BaseHead):
    """
    Head for ST-GCN model.
    Args:
        in_channels: int, input feature channels. Default: 256.
        num_classes: int, number classes. Default: 10.
    """
    def __init__(self, in_channels=256, num_classes=10, **kwargs):
        super().__init__(num_classes, in_channels, **kwargs)
        self.fcn = nn.Conv2D(in_channels=in_channels,
                             out_channels=num_classes,
                             kernel_size=1)

    def init_weights(self):
        """Initiate the parameters.
        """
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                weight_init_(layer, 'Normal', std=0.02)

    def forward(self, x):
        """Define how the head is going to run.
        """
        x = self.fcn(x)
        x = paddle.reshape_(x, (x.shape[0], -1))  # N,C,1,1 --> N,C

        return x
