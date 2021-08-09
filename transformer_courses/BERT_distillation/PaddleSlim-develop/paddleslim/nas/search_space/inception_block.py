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

__all__ = ["InceptionABlockSpace", "InceptionCBlockSpace"]
### TODO add asymmetric kernel of conv when paddle-lite support 
### inceptionB is same as inceptionA if asymmetric kernel is not support


@SEARCHSPACE.register
class InceptionABlockSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        super(InceptionABlockSpace, self).__init__(input_size, output_size,
                                                   block_num, block_mask)
        if self.block_mask == None:
            # use input_size and output_size to compute self.downsample_num
            self.downsample_num = compute_downsample_num(self.input_size,
                                                         self.output_size)
        if self.block_num != None:
            assert self.downsample_num <= self.block_num, 'downsample numeber must be LESS THAN OR EQUAL TO block_num, but NOW: downsample numeber is {}, block_num is {}'.format(
                self.downsample_num, self.block_num)

        ### self.filter_num means filter nums
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 448, 480, 512, 1024
        ])
        ### self.k_size means kernel_size
        self.k_size = np.array([3, 5])
        ### self.pool_type means pool type, 0 means avg, 1 means max
        self.pool_type = np.array([0, 1])
        ### self.repeat means repeat of 1x1 conv in branch of inception
        ### self.repeat = np.array([0,1])

    def init_tokens(self):
        """
        The initial token.
        """
        return get_random_tokens(self.range_table())

    def range_table(self):
        """
        Get range table of current search space, constrains the range of tokens.
        """
        range_table_base = []
        if self.block_mask != None:
            range_table_length = len(self.block_mask)
        else:
            range_table_length = self.block_num

        for i in range(range_table_length):
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.k_size))
            range_table_base.append(len(self.pool_type))

        return range_table_base

    def token2arch(self, tokens=None):
        """
        return net_arch function
        """
        #assert self.block_num
        if tokens is None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 9]],
                     self.filter_num[tokens[i * 9 + 1]],
                     self.filter_num[tokens[i * 9 + 2]],
                     self.filter_num[tokens[i * 9 + 3]],
                     self.filter_num[tokens[i * 9 + 4]],
                     self.filter_num[tokens[i * 9 + 5]],
                     self.filter_num[tokens[i * 9 + 6]],
                     self.k_size[tokens[i * 9 + 7]], 2 if self.block_mask == 1
                     else 1, self.pool_type[tokens[i * 9 + 8]]))
        else:
            repeat_num = int(self.block_num / self.downsample_num)
            num_minus = self.block_num % self.downsample_num
            ### if block_num > downsample_num, add stride=1 block at last (block_num-downsample_num) layers
            for i in range(self.downsample_num):
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 9]],
                     self.filter_num[tokens[i * 9 + 1]],
                     self.filter_num[tokens[i * 9 + 2]],
                     self.filter_num[tokens[i * 9 + 3]],
                     self.filter_num[tokens[i * 9 + 4]],
                     self.filter_num[tokens[i * 9 + 5]],
                     self.filter_num[tokens[i * 9 + 6]],
                     self.k_size[tokens[i * 9 + 7]], 2,
                     self.pool_type[tokens[i * 9 + 8]]))
                ### if block_num / downsample_num > 1, add (block_num / downsample_num) times stride=1 block 
                for k in range(repeat_num - 1):
                    kk = k * self.downsample_num + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[kk * 9]],
                         self.filter_num[tokens[kk * 9 + 1]],
                         self.filter_num[tokens[kk * 9 + 2]],
                         self.filter_num[tokens[kk * 9 + 3]],
                         self.filter_num[tokens[kk * 9 + 4]],
                         self.filter_num[tokens[kk * 9 + 5]],
                         self.filter_num[tokens[kk * 9 + 6]],
                         self.k_size[tokens[kk * 9 + 7]], 1,
                         self.pool_type[tokens[kk * 9 + 8]]))

                if self.downsample_num - i <= num_minus:
                    j = self.downsample_num * (repeat_num - 1) + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[j * 9]],
                         self.filter_num[tokens[j * 9 + 1]],
                         self.filter_num[tokens[j * 9 + 2]],
                         self.filter_num[tokens[j * 9 + 3]],
                         self.filter_num[tokens[j * 9 + 4]],
                         self.filter_num[tokens[j * 9 + 5]],
                         self.filter_num[tokens[j * 9 + 6]],
                         self.k_size[tokens[j * 9 + 7]], 1,
                         self.pool_type[tokens[j * 9 + 8]]))

            if self.downsample_num == 0 and self.block_num != 0:
                for i in range(len(self.block_num)):
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[i * 9]],
                         self.filter_num[tokens[i * 9 + 1]],
                         self.filter_num[tokens[i * 9 + 2]],
                         self.filter_num[tokens[i * 9 + 3]],
                         self.filter_num[tokens[i * 9 + 4]],
                         self.filter_num[tokens[i * 9 + 5]],
                         self.filter_num[tokens[i * 9 + 6]],
                         self.k_size[tokens[i * 9 + 7]], 1,
                         self.pool_type[tokens[i * 9 + 8]]))

        def net_arch(input, return_mid_layer=False, return_block=None):
            layer_count = 0
            mid_layer = dict()
            for i, layer_setting in enumerate(self.bottleneck_params_list):
                filter_nums = layer_setting[0:7]
                filter_size = layer_setting[7]
                stride = layer_setting[8]
                pool_type = 'avg' if layer_setting[9] == 0 else 'max'
                if stride == 2:
                    layer_count += 1
                if check_points((layer_count - 1), return_block):
                    mid_layer[layer_count - 1] = input

                input = self._inceptionA(
                    input,
                    A_tokens=filter_nums,
                    filter_size=int(filter_size),
                    stride=stride,
                    pool_type=pool_type,
                    name='inceptionA_{}'.format(i + 1))

            if return_mid_layer:
                return input, mid_layer
            else:
                return input,

        return net_arch

    def _inceptionA(self,
                    data,
                    A_tokens,
                    filter_size,
                    stride,
                    pool_type,
                    name=None):
        pool1 = fluid.layers.pool2d(
            input=data,
            pool_size=filter_size,
            pool_padding='SAME',
            pool_type=pool_type,
            name=name + '_pool2d')
        conv1 = conv_bn_layer(
            input=pool1,
            filter_size=1,
            num_filters=A_tokens[0],
            stride=stride,
            act='relu',
            name=name + '_conv1')

        conv2 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=A_tokens[1],
            stride=stride,
            act='relu',
            name=name + '_conv2')

        conv3 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=A_tokens[2],
            stride=1,
            act='relu',
            name=name + '_conv3_1')
        conv3 = conv_bn_layer(
            input=conv3,
            filter_size=filter_size,
            num_filters=A_tokens[3],
            stride=stride,
            act='relu',
            name=name + '_conv3_2')

        conv4 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=A_tokens[4],
            stride=1,
            act='relu',
            name=name + '_conv4_1')
        conv4 = conv_bn_layer(
            input=conv4,
            filter_size=filter_size,
            num_filters=A_tokens[5],
            stride=1,
            act='relu',
            name=name + '_conv4_2')
        conv4 = conv_bn_layer(
            input=conv4,
            filter_size=filter_size,
            num_filters=A_tokens[6],
            stride=stride,
            act='relu',
            name=name + '_conv4_3')

        concat = fluid.layers.concat(
            [conv1, conv2, conv3, conv4], axis=1, name=name + '_concat')
        return concat


@SEARCHSPACE.register
class InceptionCBlockSpace(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        super(InceptionCBlockSpace, self).__init__(input_size, output_size,
                                                   block_num, block_mask)
        if self.block_mask == None:
            # use input_size and output_size to compute self.downsample_num
            self.downsample_num = compute_downsample_num(self.input_size,
                                                         self.output_size)
        if self.block_num != None:
            assert self.downsample_num <= self.block_num, 'downsample numeber must be LESS THAN OR EQUAL TO block_num, but NOW: downsample numeber is {}, block_num is {}'.format(
                self.downsample_num, self.block_num)

        ### self.filter_num means filter nums
        self.filter_num = np.array([
            3, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 144, 160, 192, 224,
            256, 320, 384, 448, 480, 512, 1024
        ])
        ### self.k_size means kernel_size
        self.k_size = np.array([3, 5])
        ### self.pool_type means pool type, 0 means avg, 1 means max
        self.pool_type = np.array([0, 1])
        ### self.repeat means repeat of 1x1 conv in branch of inception
        ### self.repeat = np.array([0,1])

    def init_tokens(self):
        """
        The initial token.
        """
        return get_random_tokens(self.range_table())

    def range_table(self):
        """
        Get range table of current search space, constrains the range of tokens.
        """
        range_table_base = []
        if self.block_mask != None:
            range_table_length = len(self.block_mask)
        else:
            range_table_length = self.block_num

        for i in range(range_table_length):
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.filter_num))
            range_table_base.append(len(self.k_size))
            range_table_base.append(len(self.pool_type))

        return range_table_base

    def token2arch(self, tokens=None):
        """
        return net_arch function
        """
        #assert self.block_num
        if tokens is None:
            tokens = self.init_tokens()

        self.bottleneck_params_list = []
        if self.block_mask != None:
            for i in range(len(self.block_mask)):
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 11]],
                     self.filter_num[tokens[i * 11 + 1]],
                     self.filter_num[tokens[i * 11 + 2]],
                     self.filter_num[tokens[i * 11 + 3]],
                     self.filter_num[tokens[i * 11 + 4]],
                     self.filter_num[tokens[i * 11 + 5]],
                     self.filter_num[tokens[i * 11 + 6]],
                     self.filter_num[tokens[i * 11 + 7]],
                     self.filter_num[tokens[i * 11 + 8]],
                     self.k_size[tokens[i * 11 + 9]], 2 if self.block_mask == 1
                     else 1, self.pool_type[tokens[i * 11 + 10]]))
        else:
            repeat_num = int(self.block_num / self.downsample_num)
            num_minus = self.block_num % self.downsample_num
            ### if block_num > downsample_num, add stride=1 block at last (block_num-downsample_num) layers
            for i in range(self.downsample_num):
                self.bottleneck_params_list.append(
                    (self.filter_num[tokens[i * 11]],
                     self.filter_num[tokens[i * 11 + 1]],
                     self.filter_num[tokens[i * 11 + 2]],
                     self.filter_num[tokens[i * 11 + 3]],
                     self.filter_num[tokens[i * 11 + 4]],
                     self.filter_num[tokens[i * 11 + 5]],
                     self.filter_num[tokens[i * 11 + 6]],
                     self.filter_num[tokens[i * 11 + 7]],
                     self.filter_num[tokens[i * 11 + 8]],
                     self.k_size[tokens[i * 11 + 9]], 2,
                     self.pool_type[tokens[i * 11 + 10]]))
                ### if block_num / downsample_num > 1, add (block_num / downsample_num) times stride=1 block 
                for k in range(repeat_num - 1):
                    kk = k * self.downsample_num + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[kk * 11]],
                         self.filter_num[tokens[kk * 11 + 1]],
                         self.filter_num[tokens[kk * 11 + 2]],
                         self.filter_num[tokens[kk * 11 + 3]],
                         self.filter_num[tokens[kk * 11 + 4]],
                         self.filter_num[tokens[kk * 11 + 5]],
                         self.filter_num[tokens[kk * 11 + 6]],
                         self.filter_num[tokens[kk * 11 + 7]],
                         self.filter_num[tokens[kk * 11 + 8]],
                         self.k_size[tokens[kk * 11 + 9]], 1,
                         self.pool_type[tokens[kk * 11 + 10]]))

                if self.downsample_num - i <= num_minus:
                    j = self.downsample_num * (repeat_num - 1) + i
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[j * 11]],
                         self.filter_num[tokens[j * 11 + 1]],
                         self.filter_num[tokens[j * 11 + 2]],
                         self.filter_num[tokens[j * 11 + 3]],
                         self.filter_num[tokens[j * 11 + 4]],
                         self.filter_num[tokens[j * 11 + 5]],
                         self.filter_num[tokens[j * 11 + 6]],
                         self.filter_num[tokens[j * 11 + 7]],
                         self.filter_num[tokens[j * 11 + 8]],
                         self.k_size[tokens[j * 11 + 9]], 1,
                         self.pool_type[tokens[j * 11 + 10]]))

            if self.downsample_num == 0 and self.block_num != 0:
                for i in range(len(self.block_num)):
                    self.bottleneck_params_list.append(
                        (self.filter_num[tokens[i * 11]],
                         self.filter_num[tokens[i * 11 + 1]],
                         self.filter_num[tokens[i * 11 + 2]],
                         self.filter_num[tokens[i * 11 + 3]],
                         self.filter_num[tokens[i * 11 + 4]],
                         self.filter_num[tokens[i * 11 + 5]],
                         self.filter_num[tokens[i * 11 + 6]],
                         self.filter_num[tokens[i * 11 + 7]],
                         self.filter_num[tokens[i * 11 + 8]],
                         self.k_size[tokens[i * 11 + 9]], 1,
                         self.pool_type[tokens[i * 11 + 10]]))

        def net_arch(input, return_mid_layer=False, return_block=None):
            layer_count = 0
            mid_layer = dict()
            for i, layer_setting in enumerate(self.bottleneck_params_list):
                filter_nums = layer_setting[0:9]
                filter_size = layer_setting[9]
                stride = layer_setting[10]
                pool_type = 'avg' if layer_setting[11] == 0 else 'max'
                if stride == 2:
                    layer_count += 1
                if check_points((layer_count - 1), return_block):
                    mid_layer[layer_count - 1] = input

                input = self._inceptionC(
                    input,
                    C_tokens=filter_nums,
                    filter_size=int(filter_size),
                    stride=stride,
                    pool_type=pool_type,
                    name='inceptionC_{}'.format(i + 1))

            if return_mid_layer:
                return input, mid_layer
            else:
                return input,

        return net_arch

    def _inceptionC(self,
                    data,
                    C_tokens,
                    filter_size,
                    stride,
                    pool_type,
                    name=None):
        pool1 = fluid.layers.pool2d(
            input=data,
            pool_size=filter_size,
            pool_padding='SAME',
            pool_type=pool_type,
            name=name + '_pool2d')
        conv1 = conv_bn_layer(
            input=pool1,
            filter_size=1,
            num_filters=C_tokens[0],
            stride=stride,
            act='relu',
            name=name + '_conv1')

        conv2 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=C_tokens[1],
            stride=stride,
            act='relu',
            name=name + '_conv2')

        conv3 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=C_tokens[2],
            stride=1,
            act='relu',
            name=name + '_conv3_1')
        conv3_1 = conv_bn_layer(
            input=conv3,
            filter_size=filter_size,
            num_filters=C_tokens[3],
            stride=stride,
            act='relu',
            name=name + '_conv3_2_1')
        conv3_2 = conv_bn_layer(
            input=conv3,
            filter_size=filter_size,
            num_filters=C_tokens[4],
            stride=stride,
            act='relu',
            name=name + '_conv3_2_2')

        conv4 = conv_bn_layer(
            input=data,
            filter_size=1,
            num_filters=C_tokens[5],
            stride=1,
            act='relu',
            name=name + '_conv4_1')
        conv4 = conv_bn_layer(
            input=conv4,
            filter_size=filter_size,
            num_filters=C_tokens[6],
            stride=1,
            act='relu',
            name=name + '_conv4_2')
        conv4_1 = conv_bn_layer(
            input=conv4,
            filter_size=filter_size,
            num_filters=C_tokens[7],
            stride=stride,
            act='relu',
            name=name + '_conv4_3_1')
        conv4_2 = conv_bn_layer(
            input=conv4,
            filter_size=filter_size,
            num_filters=C_tokens[8],
            stride=stride,
            act='relu',
            name=name + '_conv4_3_2')

        concat = fluid.layers.concat(
            [conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2],
            axis=1,
            name=name + '_concat')
        return concat
