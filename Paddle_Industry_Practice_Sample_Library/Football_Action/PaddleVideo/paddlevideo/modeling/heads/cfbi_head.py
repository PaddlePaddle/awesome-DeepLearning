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

from .base import BaseHead
from ..registry import HEADS
from ..weight_init import weight_init_


class IA_gate(nn.Layer):
    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + paddle.tanh(a)
        a = paddle.unsqueeze(paddle.unsqueeze(a, axis=-1), axis=-1)
        x = a * x
        return x


class GCT(nn.Layer):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        x1 = paddle.zeros([1, num_channels, 1, 1])
        x2 = paddle.ones([1, num_channels, 1, 1])
        self.alpha = paddle.create_parameter(
            shape=x2.shape,
            dtype=x2.dtype,
            default_initializer=nn.initializer.Assign(x2))
        self.alpha.stop_gradient = False
        self.gamma = paddle.create_parameter(
            shape=x1.shape,
            dtype=x1.dtype,
            default_initializer=nn.initializer.Assign(x1))
        self.gamma.stop_gradient = False
        self.beta = paddle.create_parameter(
            shape=x1.shape,
            dtype=x1.dtype,
            default_initializer=nn.initializer.Assign(x1))
        self.beta.stop_gradient = False

        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = paddle.pow(
                paddle.sum(paddle.pow(x, 2), axis=[2, 3], keepdim=True) +
                self.epsilon, 0.5) * self.alpha
            norm = self.gamma / paddle.pow(
                (paddle.mean(paddle.pow(embedding, 2), axis=1, keepdim=True) +
                 self.epsilon), 0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = paddle.abs(x)
            else:
                _x = x
            embedding = paddle.sum(_x, axis=(2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (paddle.mean(
                paddle.abs(embedding), axis=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')
            exit()

        gate = 1. + paddle.tanh(embedding * norm + self.beta)

        return x * gate


class Bottleneck(nn.Layer):
    def __init__(self, inplanes, outplanes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = 4
        planes = int(outplanes / expansion)

        self.GCT1 = GCT(inplanes)
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=planes)

        self.conv2 = nn.Conv2D(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias_attr=False)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=planes)

        self.conv3 = nn.Conv2D(planes,
                               planes * expansion,
                               kernel_size=1,
                               bias_attr=False)
        self.bn3 = nn.GroupNorm(num_groups=32, num_channels=planes * expansion)
        self.relu = nn.ReLU()
        if stride != 1 or inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2D(inplanes,
                          planes * expansion,
                          kernel_size=1,
                          stride=stride,
                          bias_attr=False),
                nn.GroupNorm(num_groups=32, num_channels=planes * expansion),
            )
        else:
            downsample = None
        self.downsample = downsample

        self.stride = stride
        self.dilation = dilation

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()

    def forward(self, x):
        residual = x

        out = self.GCT1(x)
        out = self.conv1(out)
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


class _ASPPModule(nn.Layer):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.GCT = GCT(inplanes)
        self.atrous_conv = nn.Conv2D(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias_attr=False)
        self.bn = nn.GroupNorm(num_groups=int(planes / 4), num_channels=planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.GCT(x)
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data = nn.initializer.Constant(1)
                m.bias.data = nn.initializer.Constant(0)


class ASPP(nn.Layer):
    def __init__(self):
        super(ASPP, self).__init__()

        inplanes = 512
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 128,
                                 1,
                                 padding=0,
                                 dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes,
                                 128,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes,
                                 128,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes,
                                 128,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(inplanes, 128, 1, stride=1, bias_attr=False), nn.ReLU())

        self.GCT = GCT(640)
        self.conv1 = nn.Conv2D(640, 256, 1, bias_attr=False)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.relu = nn.ReLU()
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
        x = paddle.concat([x1, x2, x3, x4, x5], axis=1)

        x = self.GCT(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.initializer.KaimingNormal()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data = nn.initializer.Constant(1)
                m.bias.data = nn.initializer.Constant(0)


@HEADS.register()
class CollaborativeEnsemblerMS(nn.Layer):
    def __init__(
        self,
        model_semantic_embedding_dim=256,
        model_multi_local_distance=[[4, 8, 12, 16, 20, 24],
                                    [2, 4, 6, 8, 10, 12], [2, 4, 6, 8, 10]],
        model_head_embedding_dim=256,
        model_refine_channels=64,
        model_low_level_inplanes=256,
    ):
        super(CollaborativeEnsemblerMS, self).__init__()
        in_dim_4x = model_semantic_embedding_dim * 3 + 3 + 2 * len(
            model_multi_local_distance[0])
        in_dim_8x = model_semantic_embedding_dim * 3 + 3 + 2 * len(
            model_multi_local_distance[1])
        in_dim_16x = model_semantic_embedding_dim * 3 + 3 + 2 * len(
            model_multi_local_distance[2])
        attention_dim = model_semantic_embedding_dim * 4
        embed_dim = model_head_embedding_dim
        refine_dim = model_refine_channels
        low_level_dim = model_low_level_inplanes

        IA_in_dim = attention_dim

        self.relu = nn.ReLU()

        # stage 1

        self.S1_IA1 = IA_gate(IA_in_dim, in_dim_4x)
        self.S1_layer1 = Bottleneck(in_dim_4x, embed_dim)

        self.S1_IA2 = IA_gate(IA_in_dim, embed_dim)
        self.S1_layer2 = Bottleneck(embed_dim, embed_dim, 1, 2)

        # stage2
        self.S2_IA1 = IA_gate(IA_in_dim, embed_dim)
        self.S2_layer1 = Bottleneck(embed_dim, embed_dim * 2, 2)

        self.S2_IA2 = IA_gate(IA_in_dim, embed_dim * 2 + in_dim_8x)
        self.S2_layer2 = Bottleneck(embed_dim * 2 + in_dim_8x, embed_dim * 2, 1,
                                    2)

        self.S2_IA3 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S2_layer3 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)

        # stage3
        self.S3_IA1 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S3_layer1 = Bottleneck(embed_dim * 2, embed_dim * 2, 2)

        self.S3_IA2 = IA_gate(IA_in_dim, embed_dim * 2 + in_dim_16x)
        self.S3_layer2 = Bottleneck(embed_dim * 2 + in_dim_16x, embed_dim * 2,
                                    1, 2)

        self.S3_IA3 = IA_gate(IA_in_dim, embed_dim * 2)
        self.S3_layer3 = Bottleneck(embed_dim * 2, embed_dim * 2, 1, 4)

        self.ASPP_IA = IA_gate(IA_in_dim, embed_dim * 2)
        self.ASPP = ASPP()

        # Decoder
        self.GCT_sc = GCT(low_level_dim + embed_dim)
        self.conv_sc = nn.Conv2D(low_level_dim + embed_dim,
                                 refine_dim,
                                 1,
                                 bias_attr=False)
        self.bn_sc = nn.GroupNorm(num_groups=int(refine_dim / 4),
                                  num_channels=refine_dim)
        self.relu = nn.ReLU()

        self.IA10 = IA_gate(IA_in_dim, embed_dim + refine_dim)
        self.conv1 = nn.Conv2D(embed_dim + refine_dim,
                               int(embed_dim / 2),
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn1 = nn.GroupNorm(num_groups=32, num_channels=int(embed_dim / 2))

        self.IA11 = IA_gate(IA_in_dim, int(embed_dim / 2))
        self.conv2 = nn.Conv2D(int(embed_dim / 2),
                               int(embed_dim / 2),
                               kernel_size=3,
                               padding=1,
                               bias_attr=False)
        self.bn2 = nn.GroupNorm(num_groups=32, num_channels=int(embed_dim / 2))

        # Output
        self.IA_final_fg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)
        self.IA_final_bg = nn.Linear(IA_in_dim, int(embed_dim / 2) + 1)

        self.conv_sc.weight.data = nn.initializer.KaimingNormal()
        self.conv1.weight.data = nn.initializer.KaimingNormal()
        self.conv2.weight.data = nn.initializer.KaimingNormal()

    def forward(self, all_x, all_IA_head=None, low_level_feat=None):
        x_4x, x_8x, x_16x = all_x
        IA_head = all_IA_head[0]

        # stage 1
        x = self.S1_IA1(x_4x, IA_head)
        x = self.S1_layer1(x)

        x = self.S1_IA2(x, IA_head)
        x = self.S1_layer2(x)

        low_level_feat = paddle.concat(
            [paddle.expand(low_level_feat, [x.shape[0], -1, -1, -1]), x],
            axis=1)

        # stage 2
        x = self.S2_IA1(x, IA_head)
        x = self.S2_layer1(x)

        x = paddle.concat([x, x_8x], axis=1)
        x = self.S2_IA2(x, IA_head)
        x = self.S2_layer2(x)

        x = self.S2_IA3(x, IA_head)
        x = self.S2_layer3(x)

        # stage 3
        x = self.S3_IA1(x, IA_head)
        x = self.S3_layer1(x)

        x = paddle.concat([x, x_16x], axis=1)
        x = self.S3_IA2(x, IA_head)
        x = self.S3_layer2(x)

        x = self.S3_IA3(x, IA_head)
        x = self.S3_layer3(x)

        # ASPP + Decoder
        x = self.ASPP_IA(x, IA_head)
        x = self.ASPP(x)

        x = self.decoder(x, low_level_feat, IA_head)

        fg_logit = self.IA_logit(x, IA_head, self.IA_final_fg)
        bg_logit = self.IA_logit(x, IA_head, self.IA_final_bg)

        pred = self.augment_background_logit(fg_logit, bg_logit)

        return pred

    def IA_logit(self, x, IA_head, IA_final):
        n, c, h, w = x.shape
        x = paddle.reshape(x, [1, n * c, h, w])
        IA_output = IA_final(IA_head)
        IA_weight = IA_output[:, :c]
        IA_bias = IA_output[:, -1]
        IA_weight = paddle.reshape(IA_weight, [n, c, 1, 1])

        IA_bias = paddle.reshape(IA_bias, [-1])
        logit = paddle.reshape(
            F.conv2d(x, weight=IA_weight, bias=IA_bias, groups=n), [n, 1, h, w])
        return logit

    def decoder(self, x, low_level_feat, IA_head):
        x = F.interpolate(x,
                          size=low_level_feat.shape[2:],
                          mode='bicubic',
                          align_corners=True)

        low_level_feat = self.GCT_sc(low_level_feat)
        low_level_feat = self.conv_sc(low_level_feat)
        low_level_feat = self.bn_sc(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = paddle.concat([x, low_level_feat], axis=1)
        x = self.IA10(x, IA_head)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.IA11(x, IA_head)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

    def augment_background_logit(self, fg_logit, bg_logit):
        #  We augment the logit of absolute background by using the relative background logit of all the
        #  foreground objects.
        obj_num = fg_logit.shape[0]
        pred = fg_logit
        if obj_num > 1:
            bg_logit = bg_logit[1:obj_num, :, :, :]
            aug_bg_logit = paddle.min(bg_logit, axis=0, keepdim=True)
            pad = paddle.expand(paddle.zeros(aug_bg_logit.shape),
                                [obj_num - 1, -1, -1, -1])
            aug_bg_logit = paddle.concat([aug_bg_logit, pad], axis=0)
            pred = pred + aug_bg_logit
        pred = paddle.transpose(pred, [1, 0, 2, 3])
        return pred
