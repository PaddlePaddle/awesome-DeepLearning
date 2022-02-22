# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import UniformInitializer, ConstantInitializer
from .search_space_base import SearchSpaceBase
from .base_layer import conv_bn_layer
from .search_space_registry import SEARCHSPACE


@SEARCHSPACE.register
class DartsSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        super(DartsSpace, self).__init__(input_size, output_size, block_num,
                                         block_mask)
        self.filter_num = np.array(
            [4, 8, 12, 16, 20, 36, 54, 72, 90, 108, 144, 180, 216, 252])

    def init_tokens(self):
        return [5] * 6 + [7] * 7 + [10] * 7

    def range_table(self):
        return [len(self.filter_num)] * 20

    def token2arch(self, tokens=None):
        if tokens == None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        reduction_count = 0
        for i in range(3):
            for j in range(6):
                block_idx = i * 6 + j + reduction_count
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[block_idx]], 1))
            if i < 2:
                reduction_count += 1
                block_idx = i * 6 + j + reduction_count
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[block_idx]], 2))

        def net_arch(input, drop_prob, drop_path_mask, is_train, num_classes):
            c_in = 36
            stem_multiplier = 3
            c_curr = stem_multiplier * c_in
            x = self._conv_bn(
                input,
                c_curr,
                kernel_size=3,
                padding=1,
                stride=1,
                name='cifar10_darts_conv0')
            s0 = s1 = x

            logits_aux = None
            reduction_prev = False

            for i, layer_setting in enumerate(self.bottleneck_params_list):
                filter_num, stride = layer_setting[0], layer_setting[1]
                if stride == 2:
                    reduction = True
                else:
                    reduction = False

                if is_train:
                    drop_path_cell = drop_path_mask[:, i, :, :]
                else:
                    drop_path_cell = drop_path_mask

                s0, s1 = s1, self._cell(
                    s0,
                    s1,
                    filter_num,
                    stride,
                    reduction_prev,
                    drop_prob,
                    drop_path_cell,
                    is_train,
                    name='cifar10_darts_layer{}'.format(i + 1))
                reduction_prev = reduction

                if i == 2 * 20 // 3:
                    if is_train:
                        logits_aux = self._auxiliary_cifar(
                            s1, num_classes,
                            "cifar10_darts_/l" + str(i) + "/aux")

            logits = self._classifier(s1, num_classes, name='cifar10_darts')

            return logits, logits_aux

        return net_arch

    def _classifier(self, x, num_classes, name):
        out = fluid.layers.pool2d(x, pool_type='avg', global_pooling=True)
        out = fluid.layers.squeeze(out, axes=[2, 3])
        k = (1. / out.shape[1])**0.5
        out = fluid.layers.fc(out,
                              num_classes,
                              param_attr=fluid.ParamAttr(
                                  name=name + "/fc_weights",
                                  initializer=UniformInitializer(
                                      low=-k, high=k)),
                              bias_attr=fluid.ParamAttr(
                                  name=name + "/fc_bias",
                                  initializer=UniformInitializer(
                                      low=-k, high=k)))
        return out

    def _auxiliary_cifar(self, x, num_classes, name):
        x = fluid.layers.relu(x)
        pooled = fluid.layers.pool2d(
            x, pool_size=5, pool_stride=3, pool_padding=0, pool_type='avg')
        conv1 = self._conv_bn(
            x=pooled,
            c_out=128,
            kernel_size=1,
            padding=0,
            stride=1,
            name=name + '/conv_bn1')
        conv1 = fluid.layers.relu(conv1)
        conv2 = self._conv_bn(
            x=conv1,
            c_out=768,
            kernel_size=2,
            padding=0,
            stride=1,
            name=name + '/conv_bn2')
        conv2 = fluid.layers.relu(conv2)
        out = self._classifier(conv2, num_classes, name)
        return out

    def _cell(self,
              s0,
              s1,
              filter_num,
              stride,
              reduction_prev,
              drop_prob,
              drop_path_cell,
              is_train,
              name=None):
        if reduction_prev:
            s0 = self._factorized_reduce(s0, filter_num, name=name + '/s-2')
        else:
            s0 = self._relu_conv_bn(
                s0, filter_num, 1, 1, 0, name=name + '/s-2')
        s1 = self._relu_conv_bn(s1, filter_num, 1, 1, 0, name=name + '/s-1')

        if stride == 1:
            out = self._normal_cell(
                s0,
                s1,
                filter_num,
                drop_prob,
                drop_path_cell,
                is_train,
                name=name)
        else:
            out = self._reduction_cell(
                s0,
                s1,
                filter_num,
                drop_prob,
                drop_path_cell,
                is_train,
                name=name)
        return out

    def _normal_cell(self,
                     s0,
                     s1,
                     filter_num,
                     drop_prob,
                     drop_path_cell,
                     is_train,
                     name=None):
        hidden0_0 = self._dil_conv(
            s0,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            affine=True,
            name=name + '_normal_cell_hidden0_0')
        hidden0_1 = self._sep_conv(
            s1,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden0_1')

        if is_train:
            hidden0_0 = self._drop_path(
                hidden0_0,
                drop_prob,
                drop_path_cell[:, 0, 0],
                name=name + '_normal_cell_hidden0_0')
            hidden0_1 = self._drop_path(
                hidden0_1,
                drop_prob,
                drop_path_cell[:, 0, 1],
                name=name + '_normal_cell_hidden0_1')
        n0 = hidden0_0 + hidden0_1

        hidden1_0 = self._sep_conv(
            s0,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden1_0')
        hidden1_1 = self._sep_conv(
            s1,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden1_1')
        if is_train:
            hidden1_0 = self._drop_path(
                hidden1_0,
                drop_prob,
                drop_path_cell[:, 1, 0],
                name=name + '_normal_cell_hidden1_0')
            hidden1_1 = self._drop_path(
                hidden1_1,
                drop_prob,
                drop_path_cell[:, 1, 1],
                name=name + '_normal_cell_hidden1_1')
        n1 = hidden1_0 + hidden1_1

        hidden2_0 = self._sep_conv(
            s0,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden2_0')
        hidden2_1 = self._sep_conv(
            s1,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden2_1')
        if is_train:
            hidden2_0 = self._drop_path(
                hidden2_0,
                drop_prob,
                drop_path_cell[:, 2, 0],
                name=name + '_normal_cell_hidden2_0')
            hidden2_1 = self._drop_path(
                hidden2_1,
                drop_prob,
                drop_path_cell[:, 2, 1],
                name=name + '_normal_cell_hidden2_1')
        n2 = hidden2_0 + hidden2_1

        ### skip connect => identity
        hidden3_0 = s0
        hidden3_1 = self._sep_conv(
            s1,
            c_out=filter_num,
            kernel_size=3,
            stride=1,
            padding=1,
            affine=True,
            name=name + '_normal_cell_hidden3_1')
        if is_train:
            hidden3_1 = self._drop_path(
                hidden3_1,
                drop_prob,
                drop_path_cell[:, 3, 1],
                name=name + '_normal_cell_hidden3_1')
        n3 = hidden3_0 + hidden3_1

        out = fluid.layers.concat(
            input=[n0, n1, n2, n3], axis=1, name=name + '_normal_cell_concat')
        return out

    def _reduction_cell(self,
                        s0,
                        s1,
                        filter_num,
                        drop_prob,
                        drop_path_cell,
                        is_train,
                        name=None):
        hidden0_0 = fluid.layers.pool2d(
            input=s0,
            pool_size=3,
            pool_type="max",
            pool_stride=2,
            pool_padding=1,
            name=name + '_reduction_cell_hidden0_0')
        hidden0_1 = self._factorized_reduce(
            s1,
            filter_num,
            affine=True,
            name=name + '_reduction_cell_hidden0_1')
        if is_train:
            hidden0_0 = self._drop_path(
                hidden0_0,
                drop_prob,
                drop_path_cell[:, 0, 0],
                name=name + '_reduction_cell_hidden0_0')
        r0 = hidden0_0 + hidden0_1

        hidden1_0 = fluid.layers.pool2d(
            input=s1,
            pool_size=3,
            pool_type="max",
            pool_stride=2,
            pool_padding=1,
            name=name + '_reduction_cell_hidden1_0')
        hidden1_1 = r0
        if is_train:
            hidden1_0 = self._drop_path(
                hidden1_0,
                drop_prob,
                drop_path_cell[:, 1, 0],
                name=name + '_reduction_cell_hidden1_0')
        r1 = hidden1_0 + hidden1_1

        hidden2_0 = r0
        hidden2_1 = self._dil_conv(
            r1,
            c_out=filter_num,
            kernel_size=5,
            stride=1,
            padding=4,
            dilation=2,
            affine=True,
            name=name + '_reduction_cell_hidden2_1')
        if is_train:
            hidden2_1 = self._drop_path(
                hidden2_1,
                drop_prob,
                drop_path_cell[:, 2, 0],
                name=name + '_reduction_cell_hidden2_1')
        r2 = hidden2_0 + hidden2_1

        hidden3_0 = r0
        hidden3_1 = fluid.layers.pool2d(
            input=s1,
            pool_size=3,
            pool_type="max",
            pool_stride=2,
            pool_padding=1,
            name=name + '_reduction_cell_hidden3_1')
        if is_train:
            hidden3_1 = self._drop_path(
                hidden3_1,
                drop_prob,
                drop_path_cell[:, 3, 0],
                name=name + '_reduction_cell_hidden3_1')
        r3 = hidden3_0 + hidden3_1

        out = fluid.layers.concat(
            input=[r0, r1, r2, r3],
            axis=1,
            name=name + '_reduction_cell_concat')
        return out

    def _conv_bn(self, x, c_out, kernel_size, padding, stride, name):
        k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
        conv1 = fluid.layers.conv2d(
            x,
            c_out,
            kernel_size,
            stride=stride,
            padding=padding,
            param_attr=fluid.ParamAttr(
                name=name + "/conv",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        bn1 = fluid.layers.batch_norm(
            conv1,
            param_attr=fluid.ParamAttr(
                name=name + "/bn_scale",
                initializer=ConstantInitializer(value=1)),
            bias_attr=fluid.ParamAttr(
                name=name + "/bn_offset",
                initializer=ConstantInitializer(value=0)),
            moving_mean_name=name + "/bn_mean",
            moving_variance_name=name + "/bn_variance")
        return bn1

    def _sep_conv(self,
                  x,
                  c_out,
                  kernel_size,
                  stride,
                  padding,
                  affine=True,
                  name=''):
        c_in = x.shape[1]
        x = fluid.layers.relu(x)
        k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
        x = fluid.layers.conv2d(
            x,
            c_in,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=c_in,
            use_cudnn=False,
            param_attr=fluid.ParamAttr(
                name=name + "/sep_conv_1_1",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        k = (1. / x.shape[1] / 1 / 1)**0.5
        x = fluid.layers.conv2d(
            x,
            c_in,
            1,
            padding=0,
            param_attr=fluid.ParamAttr(
                name=name + "/sep_conv_1_2",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        gama, beta = self._bn_param_config(name, affine, "sep_conv_bn1")
        x = fluid.layers.batch_norm(
            x,
            param_attr=gama,
            bias_attr=beta,
            moving_mean_name=name + "/sep_bn1_mean",
            moving_variance_name=name + "/sep_bn1_variance")

        x = fluid.layers.relu(x)
        k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
        c_in = x.shape[1]
        x = fluid.layers.conv2d(
            x,
            c_in,
            kernel_size,
            stride=1,
            padding=padding,
            groups=c_in,
            use_cudnn=False,
            param_attr=fluid.ParamAttr(
                name=name + "/sep_conv2_1",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        k = (1. / x.shape[1] / 1 / 1)**0.5
        x = fluid.layers.conv2d(
            x,
            c_out,
            1,
            padding=0,
            param_attr=fluid.ParamAttr(
                name=name + "/sep_conv2_2",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        gama, beta = self._bn_param_config(name, affine, "sep_conv_bn2")
        x = fluid.layers.batch_norm(
            x,
            param_attr=gama,
            bias_attr=beta,
            moving_mean_name=name + "/sep_bn2_mean",
            moving_variance_name=name + "/sep_bn2_variance")
        return x

    def _dil_conv(self,
                  x,
                  c_out,
                  kernel_size,
                  stride,
                  padding,
                  dilation,
                  affine=True,
                  name=''):
        c_in = x.shape[1]
        x = fluid.layers.relu(x)
        k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
        x = fluid.layers.conv2d(
            x,
            c_in,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=c_in,
            use_cudnn=False,
            param_attr=fluid.ParamAttr(
                name=name + "/dil_conv1",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        k = (1. / x.shape[1] / 1 / 1)**0.5
        x = fluid.layers.conv2d(
            x,
            c_out,
            1,
            padding=0,
            param_attr=fluid.ParamAttr(
                name=name + "/dil_conv2",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        gama, beta = self._bn_param_config(name, affine, "dil_conv_bn")
        x = fluid.layers.batch_norm(
            x,
            param_attr=gama,
            bias_attr=beta,
            moving_mean_name=name + "/dil_bn_mean",
            moving_variance_name=name + "/dil_bn_variance")
        return x

    def _factorized_reduce(self, x, c_out, affine=True, name=''):
        assert c_out % 2 == 0
        x = fluid.layers.relu(x)
        x_sliced = x[:, :, 1:, 1:]
        k = (1. / x.shape[1] / 1 / 1)**0.5
        conv1 = fluid.layers.conv2d(
            x,
            c_out // 2,
            1,
            stride=2,
            param_attr=fluid.ParamAttr(
                name=name + "/fr_conv1",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        k = (1. / x_sliced.shape[1] / 1 / 1)**0.5
        conv2 = fluid.layers.conv2d(
            x_sliced,
            c_out // 2,
            1,
            stride=2,
            param_attr=fluid.ParamAttr(
                name=name + "/fr_conv2",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        x = fluid.layers.concat(input=[conv1, conv2], axis=1)
        gama, beta = self._bn_param_config(name, affine, "fr_bn")
        x = fluid.layers.batch_norm(
            x,
            param_attr=gama,
            bias_attr=beta,
            moving_mean_name=name + "/fr_mean",
            moving_variance_name=name + "/fr_variance")
        return x

    def _relu_conv_bn(self,
                      x,
                      c_out,
                      kernel_size,
                      stride,
                      padding,
                      affine=True,
                      name=''):
        x = fluid.layers.relu(x)
        k = (1. / x.shape[1] / kernel_size / kernel_size)**0.5
        x = fluid.layers.conv2d(
            x,
            c_out,
            kernel_size,
            stride=stride,
            padding=padding,
            param_attr=fluid.ParamAttr(
                name=name + "/rcb_conv",
                initializer=UniformInitializer(
                    low=-k, high=k)),
            bias_attr=False)
        gama, beta = self._bn_param_config(name, affine, "rcb_bn")
        x = fluid.layers.batch_norm(
            x,
            param_attr=gama,
            bias_attr=beta,
            moving_mean_name=name + "/rcb_mean",
            moving_variance_name=name + "/rcb_variance")
        return x

    def _bn_param_config(self, name='', affine=False, op=None):
        gama_name = name + "/" + str(op) + "/gama"
        beta_name = name + "/" + str(op) + "/beta"
        gama = ParamAttr(
            name=gama_name,
            initializer=ConstantInitializer(value=1),
            trainable=affine)
        beta = ParamAttr(
            name=beta_name,
            initializer=ConstantInitializer(value=0),
            trainable=affine)
        return gama, beta

    def _drop_path(self, x, drop_prob, mask, name=None):
        keep_prob = 1 - drop_prob[0]
        x = fluid.layers.elementwise_mul(
            x / keep_prob,
            mask,
            axis=0,
            name=name + '_drop_path_elementwise_mul')
        return x
