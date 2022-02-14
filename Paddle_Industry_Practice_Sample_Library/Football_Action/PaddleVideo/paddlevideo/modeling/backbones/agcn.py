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
import paddle.nn.functional as F
from ..registry import BACKBONES


class GCN(nn.Layer):
    def __init__(self, in_channels, out_channels, vertex_nums=25, stride=1):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=in_channels,
                               out_channels=3 * out_channels,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2D(in_channels=vertex_nums * 3,
                               out_channels=vertex_nums,
                               kernel_size=1)

    def forward(self, x):
        # x --- N,C,T,V
        x = self.conv1(x)  # N,3C,T,V
        N, C, T, V = x.shape
        x = paddle.reshape(x, [N, C // 3, 3, T, V])  # N,C,3,T,V
        x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])  # N,C,3,V,T
        x = paddle.reshape(x, [N, C // 3, 3 * V, T])  # N,C,3V,T
        x = paddle.transpose(x, perm=[0, 2, 1, 3])  # N,3V,C,T
        x = self.conv2(x)  # N,V,C,T
        x = paddle.transpose(x, perm=[0, 2, 3, 1])  # N,C,T,V
        return x


class Block(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 vertex_nums=25,
                 temporal_size=9,
                 stride=1,
                 residual=True):
        super(Block, self).__init__()
        self.residual = residual
        self.out_channels = out_channels

        self.bn_res = nn.BatchNorm2D(out_channels)
        self.conv_res = nn.Conv2D(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=(stride, 1))
        self.gcn = GCN(in_channels=in_channels,
                       out_channels=out_channels,
                       vertex_nums=vertex_nums)
        self.tcn = nn.Sequential(
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(temporal_size, 1),
                      padding=((temporal_size - 1) // 2, 0),
                      stride=(stride, 1)),
            nn.BatchNorm2D(out_channels),
        )

    def forward(self, x):
        if self.residual:
            y = self.conv_res(x)
            y = self.bn_res(y)
        x = self.gcn(x)
        x = self.tcn(x)
        out = x + y if self.residual else x
        out = F.relu(out)
        return out


@BACKBONES.register()
class AGCN(nn.Layer):
    """
    AGCN model improves the performance of ST-GCN using
    Adaptive Graph Convolutional Networks.
    Args:
        in_channels: int, channels of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default 2.
    """
    def __init__(self, in_channels=2, **kwargs):
        super(AGCN, self).__init__()

        self.data_bn = nn.BatchNorm1D(25 * 2)
        self.agcn = nn.Sequential(
            Block(in_channels=in_channels,
                  out_channels=64,
                  residual=False,
                  **kwargs), Block(in_channels=64, out_channels=64, **kwargs),
            Block(in_channels=64, out_channels=64, **kwargs),
            Block(in_channels=64, out_channels=64, **kwargs),
            Block(in_channels=64, out_channels=128, stride=2, **kwargs),
            Block(in_channels=128, out_channels=128, **kwargs),
            Block(in_channels=128, out_channels=128, **kwargs),
            Block(in_channels=128, out_channels=256, stride=2, **kwargs),
            Block(in_channels=256, out_channels=256, **kwargs),
            Block(in_channels=256, out_channels=256, **kwargs))

        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.shape

        x = x.transpose((0, 4, 1, 2, 3))  # N, M, C, T, V
        x = x.reshape((N * M, C, T, V))

        x = self.agcn(x)

        x = self.pool(x)  # NM,C,T,V --> NM,C,1,1
        C = x.shape[1]
        x = paddle.reshape(x, (N, M, C, 1, 1)).mean(axis=1)  # N,C,1,1

        return x
