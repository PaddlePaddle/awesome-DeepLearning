# ================================================================
#   Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import math
import datetime
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


class SlimFaceNet():
    def __init__(self, class_dim, scale=0.6, arch=None):

        assert arch is not None
        self.arch = arch
        self.class_dim = class_dim
        kernels = [3]
        expansions = [2, 4, 6]
        SE = [0, 1]
        self.table = []
        for k in kernels:
            for e in expansions:
                for se in SE:
                    self.table.append((k, e, se))

        if scale == 1.0:
            # 100% - channel
            self.Slimfacenet_bottleneck_setting = [
                # t, c , n ,s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [2, 128, 6, 1],
                [4, 128, 1, 2],
                [2, 128, 2, 1]
            ]
        elif scale == 0.9:
            # 90% - channel
            self.Slimfacenet_bottleneck_setting = [
                # t, c , n ,s
                [2, 56, 5, 2],
                [4, 116, 1, 2],
                [2, 116, 6, 1],
                [4, 116, 1, 2],
                [2, 116, 2, 1]
            ]
        elif scale == 0.75:
            # 75% - channel
            self.Slimfacenet_bottleneck_setting = [
                # t, c , n ,s
                [2, 48, 5, 2],
                [4, 96, 1, 2],
                [2, 96, 6, 1],
                [4, 96, 1, 2],
                [2, 96, 2, 1]
            ]
        elif scale == 0.6:
            # 60% - channel
            self.Slimfacenet_bottleneck_setting = [
                # t, c , n ,s
                [2, 40, 5, 2],
                [4, 76, 1, 2],
                [2, 76, 6, 1],
                [4, 76, 1, 2],
                [2, 76, 2, 1]
            ]
        else:
            print('WRONG scale')
            exit()
        self.extract_feature = True

    def set_extract_feature_flag(self, flag):
        self.extract_feature = flag

    def net(self, input, label=None):
        x = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=64,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            name='conv3x3')
        x = self.conv_bn_layer(
            x,
            filter_size=3,
            num_filters=64,
            stride=1,
            padding=1,
            num_groups=64,
            if_act=True,
            name='dw_conv3x3')

        in_c = 64
        cnt = 0
        for _exp, out_c, times, _stride in self.Slimfacenet_bottleneck_setting:
            for i in range(times):
                stride = _stride if i == 0 else 1
                filter_size, exp, se = self.table[self.arch[cnt]]
                se = False if se == 0 else True
                x = self.residual_unit(
                    x,
                    num_in_filter=in_c,
                    num_out_filter=out_c,
                    stride=stride,
                    filter_size=filter_size,
                    expansion_factor=exp,
                    use_se=se,
                    name='residual_unit' + str(cnt + 1))
                cnt += 1
                in_c = out_c

        out_c = 512
        x = self.conv_bn_layer(
            x,
            filter_size=1,
            num_filters=out_c,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name='conv1x1')
        x = self.conv_bn_layer(
            x,
            filter_size=(7, 6),
            num_filters=out_c,
            stride=1,
            padding=0,
            num_groups=out_c,
            if_act=False,
            name='global_dw_conv7x7')
        x = fluid.layers.conv2d(
            x,
            num_filters=128,
            filter_size=1,
            stride=1,
            padding=0,
            groups=1,
            act=None,
            use_cudnn=True,
            param_attr=ParamAttr(
                name='linear_conv1x1_weights',
                initializer=MSRA(),
                regularizer=fluid.regularizer.L2Decay(4e-4)),
            bias_attr=False)
        bn_name = 'linear_conv1x1_bn'
        x = fluid.layers.batch_norm(
            x,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

        x = fluid.layers.reshape(x, shape=[x.shape[0], x.shape[1]])

        if self.extract_feature:
            return x

        out = self.arc_margin_product(
            x, label, self.class_dim, s=32.0, m=0.50, mode=2)
        softmax = fluid.layers.softmax(input=out)
        cost = fluid.layers.cross_entropy(input=softmax, label=label)
        loss = fluid.layers.mean(x=cost)
        acc = fluid.layers.accuracy(input=out, label=label, k=1)
        return loss, acc

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      expansion_factor,
                      use_se=False,
                      name=None):

        num_expfilter = int(round(num_in_filter * expansion_factor))
        input_data = input

        expand_conv = self.conv_bn_layer(
            input=input,
            filter_size=1,
            num_filters=num_expfilter,
            stride=1,
            padding=0,
            if_act=True,
            name=name + '_expand')

        depthwise_conv = self.conv_bn_layer(
            input=expand_conv,
            filter_size=filter_size,
            num_filters=num_expfilter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            num_groups=num_expfilter,
            use_cudnn=True,
            name=name + '_depthwise')

        if use_se:
            depthwise_conv = self.se_block(
                input=depthwise_conv,
                num_out_filter=num_expfilter,
                name=name + '_se')

        linear_conv = self.conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return linear_conv
        else:
            return fluid.layers.elementwise_add(
                x=input_data, y=linear_conv, act=None)

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        num_mid_filter = int(num_out_filter // ratio)
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act=None,
            param_attr=ParamAttr(name=name + '_1_weights'),
            bias_attr=ParamAttr(name=name + '_1_offset'))
        conv1 = fluid.layers.prelu(
            conv1,
            mode='channel',
            param_attr=ParamAttr(
                name=name + '_prelu',
                regularizer=fluid.regularizer.L2Decay(0.0)))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(name=name + '_2_weights'),
            bias_attr=ParamAttr(name=name + '_2_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      name=None,
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                name=name + '_weights', initializer=MSRA()),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            return fluid.layers.prelu(
                bn,
                mode='channel',
                param_attr=ParamAttr(
                    name=name + '_prelu',
                    regularizer=fluid.regularizer.L2Decay(0.0)))
        else:
            return bn

    def arc_margin_product(self, input, label, out_dim, s=32.0, m=0.50,
                           mode=2):
        input_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        weight = fluid.layers.create_parameter(
            shape=[out_dim, input.shape[1]],
            dtype='float32',
            name='weight_norm',
            attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(),
                regularizer=fluid.regularizer.L2Decay(4e-4)))

        weight_norm = fluid.layers.sqrt(
            fluid.layers.reduce_sum(
                fluid.layers.square(weight), dim=1))
        weight = fluid.layers.elementwise_div(weight, weight_norm, axis=0)
        weight = fluid.layers.transpose(weight, perm=[1, 0])
        cosine = fluid.layers.mul(input, weight)
        sine = fluid.layers.sqrt(1.0 - fluid.layers.square(cosine))

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        phi = cosine * cos_m - sine * sin_m

        th = math.cos(math.pi - m)
        mm = math.sin(math.pi - m) * m

        if mode == 1:
            phi = self.paddle_where_more_than(cosine, 0, phi, cosine)
        elif mode == 2:
            phi = self.paddle_where_more_than(cosine, th, phi, cosine - mm)
        else:
            pass

        one_hot = fluid.one_hot(input=label, depth=out_dim)
        output = fluid.layers.elementwise_mul(
            one_hot, phi) + fluid.layers.elementwise_mul(
                (1.0 - one_hot), cosine)
        output = output * s
        return output

    def paddle_where_more_than(self, target, limit, x, y):
        mask = fluid.layers.cast(x=(target > limit), dtype='float32')
        output = fluid.layers.elementwise_mul(
            mask, x) + fluid.layers.elementwise_mul((1.0 - mask), y)
        return output


def SlimFaceNet_A_x0_60(class_dim=None, scale=0.6, arch=None):
    scale = 0.6
    arch = [0, 1, 5, 1, 0, 2, 1, 2, 0, 1, 2, 1, 1, 0, 1]
    return SlimFaceNet(class_dim=class_dim, scale=scale, arch=arch)


def SlimFaceNet_B_x0_75(class_dim=None, scale=0.6, arch=None):
    scale = 0.75
    arch = [1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 3, 2, 2, 3]
    return SlimFaceNet(class_dim=class_dim, scale=scale, arch=arch)


def SlimFaceNet_C_x0_75(class_dim=None, scale=0.6, arch=None):
    scale = 0.75
    arch = [1, 3, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 5, 5, 5]
    return SlimFaceNet(class_dim=class_dim, scale=scale, arch=arch)


if __name__ == "__main__":
    x = fluid.data(name='x', shape=[-1, 3, 112, 112], dtype='float32')
    print(x.shape)
    model = SlimFaceNet(10000, [1, 3, 3, 1, 1, 0, 0, 1, 0, 1, 1, 0, 5, 5, 3])
    y = model.net(x)
