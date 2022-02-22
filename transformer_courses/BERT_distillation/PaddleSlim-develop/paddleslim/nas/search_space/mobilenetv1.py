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
from .utils import check_points

__all__ = ["MobileNetV1Space"]


@SEARCHSPACE.register
class MobileNetV1Space(SearchSpaceBase):
    def __init__(self, input_size, output_size, block_num, block_mask):
        super(MobileNetV1Space, self).__init__(input_size, output_size,
                                               block_num, block_mask)
        # self.head_num means the channel of first convolution
        self.head_num = np.array([3, 4, 8, 12, 16, 24, 32])  # 7
        # self.filter_num1 ~ self.filtet_num9 means channel of the following convolution
        self.filter_num1 = np.array([3, 4, 8, 12, 16, 24, 32, 48])  # 8
        self.filter_num2 = np.array([8, 12, 16, 24, 32, 48, 64, 80])  # 8
        self.filter_num3 = np.array(
            [16, 24, 32, 48, 64, 80, 96, 128, 144, 160])  #10
        self.filter_num4 = np.array(
            [24, 32, 48, 64, 80, 96, 128, 144, 160, 192])  #10
        self.filter_num5 = np.array(
            [32, 48, 64, 80, 96, 128, 144, 160, 192, 224, 256, 320])  #12
        self.filter_num6 = np.array(
            [64, 80, 96, 128, 144, 160, 192, 224, 256, 320, 384])  #11
        self.filter_num7 = np.array([
            64, 80, 96, 128, 144, 160, 192, 224, 256, 320, 384, 512, 1024, 1048
        ])  #14
        self.filter_num8 = np.array(
            [128, 144, 160, 192, 224, 256, 320, 384, 512, 576, 640, 704,
             768])  #13
        self.filter_num9 = np.array(
            [160, 192, 224, 256, 320, 384, 512, 640, 768, 832, 1024,
             1048])  #12
        # self.k_size means kernel size
        self.k_size = np.array([3, 5])  #2
        # self.repeat means repeat_num in forth downsample 
        self.repeat = np.array([1, 2, 3, 4, 5, 6])  #6

    def init_tokens(self):
        """
        The initial token.
        The first one is the index of the first layers' channel in self.head_num,
        each line in the following represent the index of the [filter_num1, filter_num2, kernel_size]
        and depth means repeat times for forth downsample
        """
        # yapf: disable
        base_init_tokens = [6,  # 32
            6, 6, 0,  # 32, 64, 3
            6, 7, 0,  # 64, 128, 3
            7, 6, 0,  # 128, 128, 3
            6, 10, 0,  # 128, 256, 3
            10, 8, 0,  # 256, 256, 3
            8, 11, 0,  # 256, 512, 3
            4,  # depth 5
            11, 8, 0,  # 512, 512, 3
            8, 10, 0,  # 512, 1024, 3
            10, 10, 0]  # 1024, 1024, 3
        # yapf: enable
        return base_init_tokens

    def range_table(self):
        """
        Get range table of current search space, constrains the range of tokens.
        """
        # yapf: disable
        base_range_table = [len(self.head_num),
            len(self.filter_num1), len(self.filter_num2), len(self.k_size),
            len(self.filter_num2), len(self.filter_num3), len(self.k_size),
            len(self.filter_num3), len(self.filter_num4), len(self.k_size),
            len(self.filter_num4), len(self.filter_num5), len(self.k_size),
            len(self.filter_num5), len(self.filter_num6), len(self.k_size),
            len(self.filter_num6), len(self.filter_num7), len(self.k_size),
            len(self.repeat),
            len(self.filter_num7), len(self.filter_num8), len(self.k_size),
            len(self.filter_num8), len(self.filter_num9), len(self.k_size),
            len(self.filter_num9), len(self.filter_num9), len(self.k_size)]
        # yapf: enable
        return base_range_table

    def token2arch(self, tokens=None):

        if tokens is None:
            tokens = self.tokens()

        self.bottleneck_param_list = []

        # tokens[0] = 32
        # 32, 64
        self.bottleneck_param_list.append(
            (self.filter_num1[tokens[1]], self.filter_num2[tokens[2]], 1,
             self.k_size[tokens[3]]))
        # 64 128 128 128
        self.bottleneck_param_list.append(
            (self.filter_num2[tokens[4]], self.filter_num3[tokens[5]], 2,
             self.k_size[tokens[6]]))
        self.bottleneck_param_list.append(
            (self.filter_num3[tokens[7]], self.filter_num4[tokens[8]], 1,
             self.k_size[tokens[9]]))
        # 128 256 256 256
        self.bottleneck_param_list.append(
            (self.filter_num4[tokens[10]], self.filter_num5[tokens[11]], 2,
             self.k_size[tokens[12]]))
        self.bottleneck_param_list.append(
            (self.filter_num5[tokens[13]], self.filter_num6[tokens[14]], 1,
             self.k_size[tokens[15]]))
        # 256 512 (512 512) *  5
        self.bottleneck_param_list.append(
            (self.filter_num6[tokens[16]], self.filter_num7[tokens[17]], 2,
             self.k_size[tokens[18]]))
        for i in range(self.repeat[tokens[19]]):
            self.bottleneck_param_list.append(
                (self.filter_num7[tokens[20]], self.filter_num8[tokens[21]], 1,
                 self.k_size[tokens[22]]))
        # 512 1024 1024 1024
        self.bottleneck_param_list.append(
            (self.filter_num8[tokens[23]], self.filter_num9[tokens[24]], 2,
             self.k_size[tokens[25]]))
        self.bottleneck_param_list.append(
            (self.filter_num9[tokens[26]], self.filter_num9[tokens[27]], 1,
             self.k_size[tokens[28]]))

        def _modify_bottle_params(output_stride=None):
            if output_stride is not None and output_stride % 2 != 0:
                raise Exception("output stride must to be even number")
            if output_stride is None:
                return
            else:
                stride = 2
                for i, layer_setting in enumerate(self.bottleneck_params_list):
                    f1, f2, s, ks = layer_setting
                    stride = stride * s
                    if stride > output_stride:
                        s = 1
                    self.bottleneck_params_list[i] = (f1, f2, s, ks)

        def net_arch(input,
                     scale=1.0,
                     return_block=None,
                     end_points=None,
                     output_stride=None):
            self.scale = scale
            _modify_bottle_params(output_stride)

            decode_ends = dict()

            input = conv_bn_layer(
                input=input,
                filter_size=3,
                num_filters=self.head_num[tokens[0]],
                stride=2,
                name='mobilenetv1_conv1')

            layer_count = 1
            for i, layer_setting in enumerate(self.bottleneck_param_list):
                filter_num1, filter_num2, stride, kernel_size = layer_setting
                if stride == 2:
                    layer_count += 1
                ### return_block and end_points means block num
                if check_points((layer_count - 1), return_block):
                    decode_ends[layer_count - 1] = input

                if check_points((layer_count - 1), end_points):
                    return input, decode_ends
                input = self._depthwise_separable(
                    input=input,
                    num_filters1=filter_num1,
                    num_filters2=filter_num2,
                    num_groups=filter_num1,
                    stride=stride,
                    scale=self.scale,
                    kernel_size=int(kernel_size),
                    name='mobilenetv1_{}'.format(str(i + 1)))

            ### return_block and end_points means block num
            if check_points(layer_count, end_points):
                return input, decode_ends

            input = fluid.layers.pool2d(
                input=input,
                pool_type='avg',
                global_pooling=True,
                name='mobilenetv1_last_pool')

            return input

        return net_arch

    def _depthwise_separable(self,
                             input,
                             num_filters1,
                             num_filters2,
                             num_groups,
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
