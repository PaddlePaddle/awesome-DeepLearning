#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    'SlimMobileNet_v1', 'SlimMobileNet_v2', 'SlimMobileNet_v3',
    'SlimMobileNet_v4', 'SlimMobileNet_v5'
]


class SlimMobileNet():
    def __init__(self, scale=1.0, model_name='large', token=[]):
        assert len(token) >= 45
        self.kernel_size_lis = token[:20]
        self.exp_lis = token[20:40]
        self.depth_lis = token[40:45]

        self.scale = scale
        self.inplanes = 16
        if model_name == "large":
            self.cfg_channel = [16, 24, 40, 80, 112, 160]
            self.cfg_stride = [1, 2, 2, 2, 1, 2]
            self.cfg_se = [False, False, True, False, True, True]
            self.cfg_act = [
                'relu', 'relu', 'relu', 'hard_swish', 'hard_swish', 'hard_swish'
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

    def net(self, input, class_dim=1000):
        scale = self.scale
        inplanes = self.inplanes

        kernel_size_lis = self.kernel_size_lis
        exp_lis = self.exp_lis
        depth_lis = self.depth_lis
        cfg_channel = self.cfg_channel
        cfg_stride = self.cfg_stride
        cfg_se = self.cfg_se
        cfg_act = self.cfg_act

        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand
        #conv1
        conv = self.conv_bn_layer(
            input,
            filter_size=3,
            num_filters=self.make_divisible(inplanes * scale),
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        inplanes = self.make_divisible(inplanes * scale)

        #conv2
        num_mid_filter = self.make_divisible(scale * inplanes)
        _num_out_filter = cfg_channel[0]
        num_out_filter = self.make_divisible(scale * _num_out_filter)
        conv = self.residual_unit(
            input=conv,
            num_in_filter=inplanes,
            num_mid_filter=num_mid_filter,
            num_out_filter=num_out_filter,
            act=cfg_act[0],
            stride=cfg_stride[0],
            filter_size=3,
            use_se=cfg_se[0],
            name='conv2',
            short=True)
        inplanes = self.make_divisible(scale * cfg_channel[0])

        i = 3
        for depth_id in range(len(depth_lis)):
            for repeat_time in range(depth_lis[depth_id]):
                num_mid_filter = self.make_divisible(
                    scale * _num_out_filter *
                    exp_lis[depth_id * 4 + repeat_time])
                _num_out_filter = cfg_channel[depth_id + 1]
                num_out_filter = self.make_divisible(scale * _num_out_filter)
                stride = cfg_stride[depth_id + 1] if repeat_time == 0 else 1
                conv = self.residual_unit(
                    input=conv,
                    num_in_filter=inplanes,
                    num_mid_filter=num_mid_filter,
                    num_out_filter=num_out_filter,
                    act=cfg_act[depth_id + 1],
                    stride=stride,
                    filter_size=kernel_size_lis[depth_id * 4 + repeat_time],
                    use_se=cfg_se[depth_id + 1],
                    name='conv' + str(i))

                inplanes = self.make_divisible(scale *
                                               cfg_channel[depth_id + 1])
                i += 1

        conv = self.conv_bn_layer(
            input=conv,
            filter_size=1,
            num_filters=self.make_divisible(scale * cls_ch_squeeze),
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv_last')
        conv = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv = fluid.layers.conv2d(
            input=conv,
            num_filters=cls_ch_expand,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name='last_1x1_conv_weights'),
            bias_attr=False)
        conv = fluid.layers.hard_swish(conv)
        drop = fluid.layers.dropout(x=conv, dropout_prob=0.2)
        out = fluid.layers.fc(input=drop,
                              size=class_dim,
                              param_attr=ParamAttr(name='fc_weights'),
                              bias_attr=ParamAttr(name='fc_offset'))
        return out

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      act=None,
                      name=None,
                      use_cudnn=True,
                      res_last_bn_init=False):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = fluid.layers.hard_swish(bn)
        return bn

    def make_divisible(self, v, divisor=8, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        num_mid_filter = num_out_filter // ratio
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(name=name + '_1_weights'),
            bias_attr=ParamAttr(name=name + '_1_offset'))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(name=name + '_2_weights'),
            bias_attr=ParamAttr(name=name + '_2_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_mid_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      act=None,
                      use_se=False,
                      name=None,
                      short=False):

        if not short:
            conv0 = self.conv_bn_layer(
                input=input,
                filter_size=1,
                num_filters=num_mid_filter,
                stride=1,
                padding=0,
                if_act=True,
                act=act,
                name=name + '_expand')
        else:
            conv0 = input

        conv1 = self.conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=False,
            name=name + '_depthwise')
        if use_se:
            conv1 = self.se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self.conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear',
            res_last_bn_init=True)
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input, y=conv2, act=None)


def SlimMobileNet_v1(token):
    token = [
        5, 3, 3, 7, 3, 3, 5, 7, 3, 3, 3, 3, 3, 3, 7, 3, 5, 3, 3, 3, 3, 3, 3, 6,
        3, 3, 3, 3, 4, 4, 4, 6, 4, 3, 4, 3, 6, 4, 3, 3, 2, 2, 2, 2, 4
    ]
    model = SlimMobileNet(model_name='large', scale=1.0, token=token)
    return model


def SlimMobileNet_v2(token):
    token = [
        5, 3, 5, 7, 3, 3, 7, 3, 5, 3, 3, 7, 3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 4, 6,
        3, 3, 6, 3, 4, 4, 3, 4, 4, 4, 3, 6, 6, 4, 3, 3, 2, 2, 3, 2, 4
    ]
    model = SlimMobileNet(model_name='large', scale=1.0, token=token)
    return model


def SlimMobileNet_v3(token):
    token = [
        3, 3, 3, 3, 5, 3, 7, 7, 7, 3, 3, 7, 5, 3, 5, 7, 5, 3, 3, 3, 3, 3, 3, 3,
        3, 4, 3, 4, 3, 6, 4, 4, 4, 4, 6, 3, 6, 4, 6, 3, 2, 2, 3, 2, 4
    ]
    model = SlimMobileNet(model_name='large', scale=1.0, token=token)
    return model


def SlimMobileNet_v4(token):
    token = [
        3, 3, 3, 3, 5, 3, 3, 5, 7, 3, 5, 5, 5, 3, 3, 7, 3, 5, 3, 3, 3, 3, 4, 6,
        3, 4, 4, 6, 4, 6, 4, 6, 4, 6, 4, 4, 6, 6, 6, 4, 2, 3, 3, 3, 4
    ]
    model = SlimMobileNet(model_name='large', scale=1.0, token=token)
    return model


def SlimMobileNet_v5(token):
    token = [
        7, 7, 3, 5, 7, 3, 5, 3, 7, 5, 3, 3, 5, 3, 7, 5, 7, 7, 5, 3, 3, 3, 6, 3,
        4, 6, 3, 6, 6, 3, 6, 4, 6, 6, 4, 3, 6, 6, 6, 6, 4, 4, 4, 4, 4
    ]
    model = SlimMobileNet(model_name='large', scale=1.0, token=token)
    return model


if __name__ == "__main__":
    pass
