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

from ..registry import HEADS
from .base import BaseHead

import paddle
import paddle.nn.functional as F

from ..weight_init import weight_init_


@HEADS.register()
class SlowFastHead(BaseHead):
    """
    ResNe(X)t 3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """
    def __init__(self,
                 width_per_group,
                 alpha,
                 beta,
                 num_classes,
                 num_frames,
                 crop_size,
                 dropout_rate,
                 pool_size_ratio=[[1, 1, 1], [1, 1, 1]],
                 loss_cfg=dict(name='CrossEntropyLoss'),
                 multigrid_short=False,
                 **kwargs):
        """
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
        """
        super().__init__(num_classes, loss_cfg, **kwargs)
        self.multigrid_short = multigrid_short
        self.width_per_group = width_per_group
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.dropout_rate = dropout_rate
        self.pool_size_ratio = pool_size_ratio

        self.dim_in = [
            self.width_per_group * 32,
            self.width_per_group * 32 // self.beta,
        ]
        self.pool_size = [None, None] if self.multigrid_short else [
            [
                self.num_frames // self.alpha // self.pool_size_ratio[0][0],
                self.crop_size // 32 // self.pool_size_ratio[0][1],
                self.crop_size // 32 // self.pool_size_ratio[0][2],
            ],
            [
                self.num_frames // self.pool_size_ratio[1][0],
                self.crop_size // 32 // self.pool_size_ratio[1][1],
                self.crop_size // 32 // self.pool_size_ratio[1][2],
            ],
        ]

        assert (len({len(self.pool_size), len(self.dim_in)
                     }) == 1), "pathway dimensions are not consistent."
        self.num_pathways = len(self.pool_size)

        self.dropout = paddle.nn.Dropout(p=self.dropout_rate)

        self.projection = paddle.nn.Linear(
            in_features=sum(self.dim_in),
            out_features=self.num_classes,
        )

    def init_weights(self):
        weight_init_(self.projection,
                     "Normal",
                     bias_value=0.0,
                     mean=0.0,
                     std=0.01)

    def forward(self, inputs):
        assert (len(inputs) == self.num_pathways
                ), "Input tensor does not contain {} pathway".format(
                    self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            if self.pool_size[pathway] is None:
                tmp_out = F.adaptive_avg_pool3d(x=inputs[pathway],
                                                output_size=(1, 1, 1),
                                                data_format="NCDHW")
            else:
                tmp_out = F.avg_pool3d(x=inputs[pathway],
                                       kernel_size=self.pool_size[pathway],
                                       stride=1,
                                       data_format="NCDHW")
            pool_out.append(tmp_out)

        x = paddle.concat(x=pool_out, axis=1)
        x = paddle.transpose(x=x, perm=(0, 2, 3, 4, 1))

        # Perform dropout.
        if self.dropout_rate > 0.0:
            x = self.dropout(x)

        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:  # attr of base class
            x = F.softmax(x, axis=4)
            x = paddle.mean(x, axis=[1, 2, 3])

        x = paddle.reshape(x, shape=(x.shape[0], -1))
        return x
