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
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..registry import BACKBONES


class FrozenBatchNorm2D(nn.Layer):
    """
    BatchNorm2D where the batch statistics and the affine parameters
    are fixed
    """
    def __init__(self, n, epsilon=1e-5):
        super(FrozenBatchNorm2D, self).__init__()
        x1 = paddle.ones([n])
        x2 = paddle.zeros([n])
        weight = self.create_parameter(
            shape=x1.shape, default_initializer=nn.initializer.Assign(x1))
        bias = self.create_parameter(
            shape=x2.shape, default_initializer=nn.initializer.Assign(x2))
        running_mean = self.create_parameter(
            shape=x2.shape, default_initializer=nn.initializer.Assign(x2))
        running_var = self.create_parameter(
            shape=x1.shape, default_initializer=nn.initializer.Assign(x1))
        self.add_parameter('weight', weight)
        self.add_parameter('bias', bias)
        self.add_parameter('running_mean', running_mean)
        self.add_parameter('running_var', running_var)
        self.epsilon = epsilon

    def forward(self, x):
        scale = self.weight * paddle.rsqrt((self.running_var + self.epsilon))
        bias = self.bias - self.running_mean * scale
        scale = paddle.reshape(scale, [1, -1, 1, 1])
        bias = paddle.reshape(bias, [1, -1, 1, 1])
        return x * scale + bias


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias_attr=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2D(planes,
                               planes * 4,
                               kernel_size=1,
                               bias_attr=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self,
                 block,
                 layers,
                 output_stride,
                 BatchNorm,
                 pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2D(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=strides[0],
                                       dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=strides[1],
                                       dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=strides[2],
                                       dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block,
                                         512,
                                         blocks=blocks,
                                         stride=strides[3],
                                         dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        self._init_weight()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation, downsample,
                  BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      dilation=dilation,
                      BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self,
                      block,
                      planes,
                      blocks,
                      stride=1,
                      dilation=1,
                      BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  dilation=blocks[0] * dilation,
                  downsample=downsample,
                  BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(
                block(self.inplanes,
                      planes,
                      stride=1,
                      dilation=blocks[i] * dilation,
                      BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input, return_mid_level=False):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        mid_level_feat = x
        x = self.layer3(x)
        x = self.layer4(x)
        if return_mid_level:
            return x, low_level_feat, mid_level_feat
        else:
            return x, low_level_feat

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data = nn.initializer.Constant(1)
                m.bias.data = nn.initializer.Constant(0)


class _ASPPModule(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2D(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias_attr=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight_attr = nn.initializer.KaimingNormal()
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Layer):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes,
                                 256,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 256,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(inplanes, 256, 1, stride=1, bias_attr=False),
            BatchNorm(256), nn.ReLU())
        self.conv1 = nn.Conv2D(1280, 256, 1, bias_attr=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.shape[2:],
                           mode='bilinear',
                           align_corners=True)
        x = paddle.concat(x=[x1, x2, x3, x4, x5], axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data = nn.initializer.Constant(1)
                m.bias.data = nn.initializer.Constant(0)


class Decoder(nn.Layer):
    def __init__(self, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        elif backbone == 'mobilenet':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2D(low_level_inplanes, 48, 1, bias_attr=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2D(304,
                      256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False), BatchNorm(256), nn.ReLU(),
            nn.Sequential(),
            nn.Conv2D(256,
                      256,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias_attr=False), BatchNorm(256), nn.ReLU(),
            nn.Sequential())

        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x,
                          size=low_level_feat.shape[2:],
                          mode='bilinear',
                          align_corners=True)
        x = paddle.concat(x=[x, low_level_feat], axis=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data = nn.initializer.Constant(1)
                m.bias.data = nn.initializer.Constant(0)


class DeepLab(nn.Layer):
    """DeepLab model for segmentation"""
    def __init__(self, backbone='resnet', output_stride=16, freeze_bn=True):
        super(DeepLab, self).__init__()

        if freeze_bn == True:
            print("Use frozen BN in DeepLab!")
            BatchNorm = FrozenBatchNorm2D
        else:
            BatchNorm = nn.BatchNorm2D

        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3],
                               output_stride,
                               BatchNorm,
                               pretrained=True)
        self.aspp = ASPP(backbone, output_stride, BatchNorm)
        self.decoder = Decoder(backbone, BatchNorm)

    def forward(self, input, return_aspp=False):
        """forward function"""
        if return_aspp:
            x, low_level_feat, mid_level_feat = self.backbone(input, True)
        else:
            x, low_level_feat = self.backbone(input)
        aspp_x = self.aspp(x)
        x = self.decoder(aspp_x, low_level_feat)

        if return_aspp:
            return x, aspp_x, low_level_feat, mid_level_feat
        else:
            return x, low_level_feat
