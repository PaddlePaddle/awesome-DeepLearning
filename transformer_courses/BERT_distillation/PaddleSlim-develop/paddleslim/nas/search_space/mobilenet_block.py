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
from .search_space_base import SearchSpaceBase
from .base_layer import conv_bn_layer
from .search_space_registry import SEARCHSPACE
from .utils import compute_downsample_num, check_points, get_random_tokens

__all__ = ["MobileNetV1BlockSpace", "MobileNetV2BlockSpace"]


@SEARCHSPACE.register
class MobileNetV2BlockSpace(SearchSpaceBase):
    def __init__(self,
                 input_size,
                 output_size,
                 block_num,
                 block_mask=None,
                 scale=1.0):
        super(MobileNetV2BlockSpace, self).__init__(input_size, output_size,
                                                    block_num, block_mask)

        if self.block_mask == None:
            # use input_size and output_size to compute self.downsample_num
            self.downsample_num = compute_downsample_num(self.input_size,
                                                         self.output_size)
        if self.block_num != None:
            assert self.downsample_num <= self.block_num, 'downsample numeber must be LESS THAN OR EQUAL TO block_num, but NOW: downsample numeber is {}, block_num is {}'.format(
                self.downsample_num, self.block_num)

        # self.filter_num means channel number
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 512
        ])  # 20
        # self.k_size means kernel size
        self.k_size = np.array([3, 5])  #2
        # self.multiply means expansion_factor of each _inverted_residual_unit
        self.multiply = np.array([1, 2, 3, 4, 5, 6])  #6
        # self.repeat means repeat_num _inverted_residual_unit in each _invresi_blocks
        self.repeat = np.array([1, 2, 3, 4, 5, 6])  #6
        self.scale = scale

    def init_tokens(self):
        return get_random_tokens(self.range_table())

    def range_table(self):
        range_table_base = []
        if self.block_mask != None:
            range_table_length = len(self.block_mask)
        else:
            range_table_length = self.block_num

        for i in range(range_table_length):
            range_table_base.append(len(self.multiply))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.repeat))
            range_table_base.append(len(self.k_size))

        return range_table_base

    def token2arch(self, tokens=None):
        """
        return mobilenetv2 net_arch function
        """

        if tokens == None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                self.bottleneck_params_list.append(
                    (self.multiply[tokens[i * 4]],
                     self.filter_num[tokens[i * 4 + 1]],
                     self.repeat[tokens[i * 4 + 2]], 2
                     if self.block_mask[i] == 1 else 1,
                     self.k_size[tokens[i * 4 + 3]]))
        else:
            repeat_num = int(self.block_num / self.downsample_num)
            num_minus = self.block_num % self.downsample_num
            ### if block_num > downsample_num, add stride=1 block at last (block_num-downsample_num) layers
            for i in range(self.downsample_num):
                self.bottleneck_params_list.append(
                    (self.multiply[tokens[i * 4]],
                     self.filter_num[tokens[i * 4 + 1]],
                     self.repeat[tokens[i * 4 + 2]], 2,
                     self.k_size[tokens[i * 4 + 3]]))

                ### if block_num / downsample_num > 1, add (block_num / downsample_num) times stride=1 block 
                for k in range(repeat_num - 1):
                    kk = k * self.downsample_num + i
                    self.bottleneck_params_list.append(
                        (self.multiply[tokens[kk * 4]],
                         self.filter_num[tokens[kk * 4 + 1]],
                         self.repeat[tokens[kk * 4 + 2]], 1,
                         self.k_size[tokens[kk * 4 + 3]]))

                if self.downsample_num - i <= num_minus:
                    j = self.downsample_num * (repeat_num - 1) + i
                    self.bottleneck_params_list.append(
                        (self.multiply[tokens[j * 4]],
                         self.filter_num[tokens[j * 4 + 1]],
                         self.repeat[tokens[j * 4 + 2]], 1,
                         self.k_size[tokens[j * 4 + 3]]))

            if self.downsample_num == 0 and self.block_num != 0:
                for i in range(len(self.block_num)):
                    self.bottleneck_params_list.append(
                        (self.multiply[tokens[i * 4]],
                         self.filter_num[tokens[i * 4 + 1]],
                         self.repeat[tokens[i * 4 + 2]], 1,
                         self.k_size[tokens[i * 4 + 3]]))

        def net_arch(input, return_mid_layer=False, return_block=None):
            # all padding is 'SAME' in the conv2d, can compute the actual padding automatic. 
            # bottleneck sequences
            in_c = int(32 * self.scale)
            mid_layer = dict()
            layer_count = 0
            depthwise_conv = None

            for i, layer_setting in enumerate(self.bottleneck_params_list):
                t, c, n, s, k = layer_setting

                if s == 2:
                    layer_count += 1
                if check_points((layer_count - 1), return_block):
                    mid_layer[layer_count - 1] = depthwise_conv

                input, depthwise_conv = self._invresi_blocks(
                    input=input,
                    in_c=in_c,
                    t=t,
                    c=int(c * self.scale),
                    n=n,
                    s=s,
                    k=int(k),
                    name='mobilenetv2_' + str(i + 1))
                in_c = int(c * self.scale)

            if check_points(layer_count, return_block):
                mid_layer[layer_count] = depthwise_conv

            if return_mid_layer:
                return input, mid_layer
            else:
                return input,

        return net_arch

    def _shortcut(self, input, data_residual):
        """Build shortcut layer.
        Args:
            input(Variable): input.
            data_residual(Variable): residual layer.
        Returns:
            Variable, layer output.
        """
        return fluid.layers.elementwise_add(input, data_residual)

    def _inverted_residual_unit(self,
                                input,
                                num_in_filter,
                                num_filters,
                                ifshortcut,
                                stride,
                                filter_size,
                                expansion_factor,
                                reduction_ratio=4,
                                name=None):
        """Build inverted residual unit.
        Args:
            input(Variable), input.
            num_in_filter(int), number of in filters.
            num_filters(int), number of filters.
            ifshortcut(bool), whether using shortcut.
            stride(int), stride.
            filter_size(int), filter size.
            padding(str|int|list), padding.
            expansion_factor(float), expansion factor.
            name(str), name.
        Returns:
            Variable, layers output.
        """
        num_expfilter = int(round(num_in_filter * expansion_factor))
        channel_expand = conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act='relu6',
            name=name + '_expand')

        bottleneck_conv = conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding='SAME',
            num_groups=num_expfilter,
            act='relu6',
            name=name + '_dwise',
            use_cudnn=False)

        depthwise_output = bottleneck_conv

        linear_out = conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding='SAME',
            num_groups=1,
            act=None,
            name=name + '_linear')
        out = linear_out
        if ifshortcut:
            out = self._shortcut(input=input, data_residual=out)
        return out, depthwise_output

    def _invresi_blocks(self, input, in_c, t, c, n, s, k, name=None):
        """Build inverted residual blocks.
        Args:
            input: Variable, input.
            in_c: int, number of in filters.
            t: float, expansion factor.
            c: int, number of filters.
            n: int, number of layers.
            s: int, stride.
            k: int, filter size.
            name: str, name.
        Returns:
            Variable, layers output.
        """
        first_block, depthwise_output = self._inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=k,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block, depthwise_output = self._inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=k,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block, depthwise_output


@SEARCHSPACE.register
class MobileNetV1BlockSpace(SearchSpaceBase):
    def __init__(self,
                 input_size,
                 output_size,
                 block_num,
                 block_mask=None,
                 scale=1.0):
        super(MobileNetV1BlockSpace, self).__init__(input_size, output_size,
                                                    block_num, block_mask)

        if self.block_mask == None:
            # use input_size and output_size to compute self.downsample_num
            self.downsample_num = compute_downsample_num(self.input_size,
                                                         self.output_size)
        if self.block_num != None:
            assert self.downsample_num <= self.block_num, 'downsample numeber must be LESS THAN OR EQUAL TO block_num, but NOW: downsample numeber is {}, block_num is {}'.format(
                self.downsample_num, self.block_num)

        # self.filter_num means channel number
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 512, 576, 640, 768, 1024, 1048
        ])
        self.k_size = np.array([3, 5])
        self.scale = scale

    def init_tokens(self):
        return get_random_tokens(self.range_table())

    def range_table(self):
        range_table_base = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                range_table_base.append(len(self.filter_num))
                range_table_base.append(len(self.filter_num))
                range_table_base.append(len(self.k_size))
        else:
            for i in range(self.block_num):
                range_table_base.append(len(self.filter_num))
                range_table_base.append(len(self.filter_num))
                range_table_base.append(len(self.k_size))

        return range_table_base

    def token2arch(self, tokens=None):
        if tokens == None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 3]],
                     self.filter_num[tokens[i * 3 + 1]], 2
                     if self.block_mask[i] == 1 else 1,
                     self.k_size[tokens[i * 3 + 2]]))
        else:
            repeat_num = int(self.block_num / self.downsample_num)
            num_minus = self.block_num % self.downsample_num
            for i in range(self.downsample_num):
                ### if block_num > downsample_num, add stride=1 block at last (block_num-downsample_num) layers
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 3]],
                     self.filter_num[tokens[i * 3 + 1]], 2,
                     self.k_size[tokens[i * 3 + 2]]))

                ### if block_num / downsample_num > 1, add (block_num / downsample_num) times stride=1 block 
                for k in range(repeat_num - 1):
                    kk = k * self.downsample_num + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[kk * 3]],
                         self.filter_num[tokens[kk * 3 + 1]], 1,
                         self.k_size[tokens[kk * 3 + 2]]))

                if self.downsample_num - i <= num_minus:
                    j = self.downsample_num * (repeat_num - 1) + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[j * 3]],
                         self.filter_num[tokens[j * 3 + 1]], 1,
                         self.k_size[tokens[j * 3 + 2]]))

            if self.downsample_num == 0 and self.block_num != 0:
                for i in range(len(self.block_num)):
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[i * 3]],
                         self.filter_num[tokens[i * 3 + 1]], 1,
                         self.k_size[tokens[i * 3 + 2]]))

        def net_arch(input, return_mid_layer=False, return_block=None):
            mid_layer = dict()
            layer_count = 0

            for i, layer_setting in enumerate(self.bottleneck_params_list):
                filter_num1, filter_num2, stride, kernel_size = layer_setting
                if stride == 2:
                    layer_count += 1
                if check_points((layer_count - 1), return_block):
                    mid_layer[layer_count - 1] = input

                input = self._depthwise_separable(
                    input=input,
                    num_filters1=filter_num1,
                    num_filters2=filter_num2,
                    stride=stride,
                    scale=self.scale,
                    kernel_size=int(kernel_size),
                    name='mobilenetv1_{}'.format(str(i + 1)))

            if return_mid_layer:
                return input, mid_layer
            else:
                return input,

        return net_arch

    def _depthwise_separable(self,
                             input,
                             num_filters1,
                             num_filters2,
                             stride,
                             scale,
                             kernel_size,
                             name=None):
        num_groups = input.shape[1]

        s_oc = int(num_filters1 * scale)
        if s_oc > num_groups:
            output_channel = s_oc - (s_oc % num_groups)
        else:
            output_channel = num_groups

        depthwise_conv = conv_bn_layer(
            input=input,
            filter_size=kernel_size,
            num_filters=output_channel,
            stride=stride,
            num_groups=num_groups,
            use_cudnn=False,
            name=name + '_dw')
        pointwise_conv = conv_bn_layer(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            name=name + '_sep')

        return pointwise_conv
