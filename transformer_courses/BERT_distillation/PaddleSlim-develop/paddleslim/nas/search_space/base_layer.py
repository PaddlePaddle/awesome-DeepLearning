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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride=1,
                  padding='SAME',
                  num_groups=1,
                  act=None,
                  name=None,
                  use_cudnn=True):
    """Build convolution and batch normalization layers.
    Args:
        input(Variable): input.
        filter_size(int): filter size.
        num_filters(int): number of filters.
        stride(int): stride.
        padding(int|list|str): padding.
        num_groups(int): number of groups.
        act(str): activation type.
        name(str): name.
        use_cudnn(bool): whether use cudnn.
    Returns:
        Variable, layers output.
    """
    conv = fluid.layers.conv2d(
        input,
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
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(name=bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance')
