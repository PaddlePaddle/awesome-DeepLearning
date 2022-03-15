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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay
from paddle.nn import Conv2D, BatchNorm
from paddle.nn import MaxPool2D, AvgPool2D

from ..registry import BACKBONES
from ..weight_init import weight_init_
from ...utils import load_ckpt

__all__ = ["ResNetTweaksTSN"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 is_tweaks_mode=False,
                 act=None,
                 lr_mult=1.0,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.is_tweaks_mode = is_tweaks_mode
        self._pool2d_avg = AvgPool2D(kernel_size=2,
                                     stride=2,
                                     padding=0,
                                     ceil_mode=True)
        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            weight_attr=ParamAttr(name=name + "_weights",
                                                  learning_rate=lr_mult),
                            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        self._batch_norm = BatchNorm(
            out_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale',
                                 learning_rate=lr_mult,
                                 regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(bn_name + '_offset',
                                learning_rate=lr_mult,
                                regularizer=L2Decay(0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_tweaks_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 act='relu',
                                 lr_mult=lr_mult,
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act='relu',
                                 lr_mult=lr_mult,
                                 name=name + "_branch2b")
        self.conv2 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels * 4,
                                 kernel_size=1,
                                 act=None,
                                 lr_mult=lr_mult,
                                 name=name + "_branch2c")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels * 4,
                                     kernel_size=1,
                                     stride=1,
                                     is_tweaks_mode=False if if_first else True,
                                     lr_mult=lr_mult,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=stride,
                                 act='relu',
                                 lr_mult=lr_mult,
                                 name=name + "_branch2a")
        self.conv1 = ConvBNLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 act=None,
                                 lr_mult=lr_mult,
                                 name=name + "_branch2b")

        if not shortcut:
            self.short = ConvBNLayer(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=1,
                                     stride=1,
                                     is_tweaks_mode=False if if_first else True,
                                     lr_mult=lr_mult,
                                     name=name + "_branch1")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y


@BACKBONES.register()
class ResNetTweaksTSN(nn.Layer):
    """ResNetTweaksTSN backbone.

    Args:
        depth (int): Depth of resnet model.
        pretrained (str): pretrained model. Default: None.
    """
    def __init__(self,
                 layers=50,
                 pretrained=None,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(ResNetTweaksTSN, self).__init__()

        self.pretrained = pretrained
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        self.lr_mult_list = lr_mult_list
        assert isinstance(
            self.lr_mult_list,
            (list, tuple
             )), "lr_mult_list should be in (list, tuple) but got {}".format(
                 type(self.lr_mult_list))
        assert len(
            self.lr_mult_list
        ) == 5, "lr_mult_list length should should be 5 but got {}".format(
            len(self.lr_mult_list))

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(in_channels=3,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=2,
                                   act='relu',
                                   lr_mult=self.lr_mult_list[0],
                                   name="conv1_1")
        self.conv1_2 = ConvBNLayer(in_channels=32,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   act='relu',
                                   lr_mult=self.lr_mult_list[0],
                                   name="conv1_2")
        self.conv1_3 = ConvBNLayer(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   act='relu',
                                   lr_mult=self.lr_mult_list[0],
                                   name="conv1_3")
        self.pool2d_max = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152, 200] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            lr_mult=self.lr_mult_list[block + 1],
                            name=conv_name))
                    self.block_list.append(bottleneck_block)
                    shortcut = True
        else:
            for block in range(len(depth)):
                shortcut = False
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(in_channels=num_channels[block]
                                   if i == 0 else num_filters[block],
                                   out_channels=num_filters[block],
                                   stride=2 if i == 0 and block != 0 else 1,
                                   shortcut=shortcut,
                                   if_first=block == i == 0,
                                   name=conv_name,
                                   lr_mult=self.lr_mult_list[block + 1]))
                    self.block_list.append(basic_block)
                    shortcut = True

    def init_weights(self):
        """Initiate the parameters.
        Note:
            1. when indicate pretrained loading path, will load it to initiate backbone.
            2. when not indicating pretrained loading path, will follow specific initialization initiate backbone. Always, Conv2D layer will be
            initiated by KaimingNormal function, and BatchNorm2d will be initiated by Constant function.
            Please refer to https://www.paddlepaddle.org.cn/documentation/docs/en/develop/api/paddle/nn/initializer/kaiming/KaimingNormal_en.html
        """
        # XXX: check bias!!! check pretrained!!!

        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, nn.Conv2D):
                    # XXX: no bias
                    weight_init_(layer, 'KaimingNormal')
                elif isinstance(layer, nn.BatchNorm2D):
                    weight_init_(layer, 'Constant', value=1)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        return y
