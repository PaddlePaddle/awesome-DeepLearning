# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import KaimingUniform
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec


import paddle.nn.functional as F
# attention

# SGE attention
class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias_attr=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_planes, epsilon=1e-5, momentum=0.01, weight_attr=False, bias_attr=False) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Layer):
    def forward(self, x):
        return paddle.concat((paddle.max(x,1).unsqueeze(1), paddle.mean(x,1).unsqueeze(1)), axis=1)


class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        print(f'************************************ use SpatialGate ************************************')

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


# used by SANN_Attention
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)
        self.bn = nn.BatchNorm2D(c2)
        self.act = nn.LeakyReLU(0.1) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SANN_Attention(nn.Layer):
    def __init__(self, k_size = 3, ch = 64, s_state = False, c_state = False):
        super(SANN_Attention, self).__init__()
        print(f'************************************use SANN_Attention s_state => {s_state} -- c_state => {c_state}')
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveAvgPool2D(1)
        self.sigmoid = nn.Sigmoid()
        self.s_state = s_state
        self.c_state = c_state

        if c_state:
            self.c_attention = nn.Sequential(nn.Conv1D(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias_attr=False),
                                                  nn.LayerNorm([1, ch]),
                                                  nn.LeakyReLU(0.3),
                                                  nn.Linear(ch, ch, bias_attr=False))

        if s_state:
            self.conv_s = nn.Sequential(Conv(ch, ch // 4, k=1))
            self.s_attention = nn.Conv2D(2, 1, 7, padding=3, bias_attr=False)

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.shape

        # channel_attention
        if self.c_state:
            y_avg = self.avg_pool(x)
            y_max = self.max_pool(x)
            y_c = self.c_attention(y_avg.squeeze(-1).transpose((0,2,1))).transpose((0,2,1)).unsqueeze(-1)+\
                  self.c_attention(y_max.squeeze(-1).transpose((0,2,1))).transpose((0,2,1)).unsqueeze(-1)
            y_c = self.sigmoid(y_c)

        #spatial_attention
        if self.s_state:
            x_s = self.conv_s(x)
            avg_out = paddle.mean(x_s, axis=1, keepdim=True)
            max_out = paddle.max(x_s, axis=1, keepdim=True)
            y_s = paddle.concat([avg_out, max_out], axis=1)
            y_s = self.sigmoid(self.s_attention(y_s))

        if self.c_state and self.s_state:
            y = x * y_s * y_c + x
        elif self.c_state:
            y = x * y_c + x
        elif self.s_state:
            y = x * y_s + x
        else:
            y = x
        return y


def fill_up_weights(up):
    weight = up.weight
    f = math.ceil(weight.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(weight.shape[2]):
        for j in range(weight.shape[3]):
            weight[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, weight.shape[0]):
        weight[c, 0, :, :] = weight[0, 0, :, :]


class IDAUp(nn.Layer):
    def __init__(self, ch_ins, ch_out, up_strides, dcn_v2=True):
        super(IDAUp, self).__init__()
        for i in range(1, len(ch_ins)):
            ch_in = ch_ins[i]
            up_s = int(up_strides[i])
            proj = nn.Sequential(
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    filter_size=3,
                    stride=1,
                    use_dcn=dcn_v2,
                    bias_on=dcn_v2,
                    norm_decay=None,
                    dcn_lr_scale=1.,
                    dcn_regularizer=None),
                nn.ReLU())
            node = nn.Sequential(
                ConvNormLayer(
                    ch_out,
                    ch_out,
                    filter_size=3,
                    stride=1,
                    use_dcn=dcn_v2,
                    bias_on=dcn_v2,
                    norm_decay=None,
                    dcn_lr_scale=1.,
                    dcn_regularizer=None),
                nn.ReLU())

            param_attr = paddle.ParamAttr(initializer=KaimingUniform())
            up = nn.Conv2DTranspose(
                ch_out,
                ch_out,
                kernel_size=up_s * 2,
                weight_attr=param_attr,
                stride=up_s,
                padding=up_s // 2,
                groups=ch_out,
                bias_attr=False)
            # TODO: uncomment fill_up_weights
            #fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, inputs, start_level, end_level):
        for i in range(start_level + 1, end_level):
            upsample = getattr(self, 'up_' + str(i - start_level))
            project = getattr(self, 'proj_' + str(i - start_level))

            inputs[i] = project(inputs[i])
            inputs[i] = upsample(inputs[i])
            node = getattr(self, 'node_' + str(i - start_level))
            inputs[i] = node(paddle.add(inputs[i], inputs[i - 1]))


class DLAUp(nn.Layer):
    def __init__(self, start_level, channels, scales, ch_in=None, dcn_v2=True):
        super(DLAUp, self).__init__()
        self.start_level = start_level
        if ch_in is None:
            ch_in = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                'ida_{}'.format(i),
                IDAUp(
                    ch_in[j:],
                    channels[j],
                    scales[j:] // scales[j],
                    dcn_v2=dcn_v2))
            scales[j + 1:] = scales[j]
            ch_in[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, inputs):
        out = [inputs[-1]]  # start with 32
        for i in range(len(inputs) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(inputs, len(inputs) - i - 2, len(inputs))
            out.insert(0, inputs[-1])
        return out


@register
@serializable
class CenterNetDLAFPN(nn.Layer):
    """
    Args:
        in_channels (list): number of input feature channels from backbone.
            [16, 32, 64, 128, 256, 512] by default, means the channels of DLA-34
        down_ratio (int): the down ratio from images to heatmap, 4 by default
        last_level (int): the last level of input feature fed into the upsamplng block
        out_channel (int): the channel of the output feature, 0 by default means
            the channel of the input feature whose down ratio is `down_ratio`
        dcn_v2 (bool): whether use the DCNv2, true by default
        
    """

    def __init__(self,
                 in_channels,
                 down_ratio=4,
                 last_level=5,
                 out_channel=0,
                 dcn_v2=True):
        super(CenterNetDLAFPN, self).__init__()
        self.first_level = int(np.log2(down_ratio))
        self.down_ratio = down_ratio
        self.last_level = last_level
        scales = [2**i for i in range(len(in_channels[self.first_level:]))]
        self.dla_up = DLAUp(
            self.first_level,
            in_channels[self.first_level:],
            scales,
            dcn_v2=dcn_v2)
        self.out_channel = out_channel
        if out_channel == 0:
            self.out_channel = in_channels[self.first_level]
        self.ida_up = IDAUp(
            in_channels[self.first_level:self.last_level],
            self.out_channel,
            [2**i for i in range(self.last_level - self.first_level)],
            dcn_v2=dcn_v2)

        self.attention = SpatialGate()
        #self.attention = SANN_Attention(c_state = False, s_state = True) # spatial_attention

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape]}

    def forward(self, body_feats):
        dla_up_feats = self.dla_up(body_feats)

        ida_up_feats = []
        for i in range(self.last_level - self.first_level):
            ida_up_feats.append(dla_up_feats[i].clone())

        self.ida_up(ida_up_feats, 0, len(ida_up_feats))

        feat = ida_up_feats[-1]
        feat = self.attention(feat)
        return feat

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.out_channel, stride=self.down_ratio)]
