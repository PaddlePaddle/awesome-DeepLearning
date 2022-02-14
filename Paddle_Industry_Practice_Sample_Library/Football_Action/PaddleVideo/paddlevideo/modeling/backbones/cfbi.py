# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import BACKBONES
from .deeplab import DeepLab


class FPN(nn.Layer):
    """FPN Layer"""
    def __init__(self, in_dim_4x, in_dim_8x, in_dim_16x, out_dim):
        super(FPN, self).__init__()
        self.toplayer = self._make_layer(in_dim_16x, out_dim)
        self.latlayer1 = self._make_layer(in_dim_8x, out_dim)
        self.latlayer2 = self._make_layer(in_dim_4x, out_dim)

        self.smooth1 = self._make_layer(out_dim,
                                        out_dim,
                                        kernel_size=3,
                                        padding=1)
        self.smooth2 = self._make_layer(out_dim,
                                        out_dim,
                                        kernel_size=3,
                                        padding=1)

    def _make_layer(self, in_dim, out_dim, kernel_size=1, padding=0):
        return nn.Sequential(
            nn.Conv2D(in_dim,
                      out_dim,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding,
                      bias_attr=False),
            nn.GroupNorm(num_groups=32, num_channels=out_dim))

    def forward(self, x_4x, x_8x, x_16x):
        """ forward function"""
        x_16x = self.toplayer(x_16x)
        x_8x = self.latlayer1(x_8x)
        x_4x = self.latlayer2(x_4x)

        x_8x = x_8x + F.interpolate(
            x_16x, size=x_8x.shape[-2:], mode='bilinear', align_corners=True)
        x_4x = x_4x + F.interpolate(
            x_8x, size=x_4x.shape[-2:], mode='bilinear', align_corners=True)

        x_8x = self.smooth1(x_8x)
        x_4x = self.smooth2(x_4x)

        return F.relu(x_4x), F.relu(x_8x), F.relu(x_16x)


@BACKBONES.register()
class CFBI(nn.Layer):
    """CFBI plus backbone"""
    def __init__(self,
                 backbone='resnet',
                 freeze_bn=True,
                 model_aspp_outdim=256,
                 in_dim_8x=512,
                 model_semantic_embedding_dim=256):  #,epsilon=1e-05):
        super(CFBI, self).__init__()
        #self.epsilon = epsilon
        self.feature_extracter = DeepLab(backbone=backbone, freeze_bn=freeze_bn)
        self.fpn = FPN(in_dim_4x=model_aspp_outdim,
                       in_dim_8x=in_dim_8x,
                       in_dim_16x=model_aspp_outdim,
                       out_dim=model_semantic_embedding_dim)

    def forward(self, x):
        """forward function"""
        x, aspp_x, low_level, mid_level = self.feature_extracter(x, True)
        x_4x, x_8x, x_16x = self.fpn(x, mid_level, aspp_x)
        return x_4x, x_8x, x_16x, low_level
