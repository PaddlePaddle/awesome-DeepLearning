# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

### NOTE: the API of this file is based on Paddle2.0, the API in layers_old.py is based on Paddle1.8

import numpy as np
import logging
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.core as core

from ...common import get_logger
from .utils.utils import compute_start_end, get_same_padding, convert_to_list
from .layers_base import *

__all__ = [
    'SuperConv2D', 'SuperConv2DTranspose', 'SuperSeparableConv2D',
    'SuperBatchNorm2D', 'SuperLinear', 'SuperInstanceNorm2D',
    'SuperGroupConv2D', 'SuperDepthwiseConv2D', 'SuperGroupConv2DTranspose',
    'SuperDepthwiseConv2DTranspose', 'SuperLayerNorm', 'SuperEmbedding',
    'SuperSyncBatchNorm'
]

_logger = get_logger(__name__, level=logging.INFO)

### TODO: if task is elastic width, need to add re_organize_middle_weight in 1x1 conv in MBBlock


class SuperConv2D(nn.Conv2D):
    """
    This interface is used to construct a callable object of the ``SuperConv2D``  class.

    Note: the channel in config need to less than first defined.

    The super convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    the feature map, H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.
    For each input :math:`X`, the equation is:
    .. math::

        Out = sigma (W \\ast X + b)

    Where:
    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:
        - Input:
          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`
          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`
        - Output:
          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        .. math::

            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1   

            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1

    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of filter. It is as same as the output
            feature map.
        filter_size (int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        candidate_config(dict, optional): Dictionary descripts candidate config of this layer,
            such as {'kernel_size': (3, 5, 7), 'channel': (4, 6, 8)}, means the kernel size of 
            this layer can be choose from (3, 5, 7), the key of candidate_config
            only can be 'kernel_size', 'channel' and 'expand_ratio', 'channel' and 'expand_ratio'
            CANNOT be set at the same time. Default: None.
        transform_kernel(bool, optional): Whether to use transform matrix to transform a large filter
            to a small filter. Default: False.
        stride (int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding (int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        dilation (int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups (int, optional): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    
    Raises:
        ValueError: if ``use_cudnn`` is not a bool value.
    Examples:
        .. code-block:: python
          import paddle 
          from paddleslim.nas.ofa.layers import SuperConv2D
          import numpy as np
          data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
          super_conv2d = SuperConv2D(3, 10, 3)
          config = {'channel': 5}
          data = paddle.to_tensor(data)
          conv = super_conv2d(data, config)

    """

    ### NOTE: filter_size, num_channels and num_filters must be the max of candidate to define a largest network.
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 candidate_config={},
                 transform_kernel=False,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW'):
        super(SuperConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

        self.candidate_config = candidate_config
        if len(candidate_config.items()) != 0:
            for k, v in candidate_config.items():
                candidate_config[k] = list(set(v))

        self.ks_set = candidate_config[
            'kernel_size'] if 'kernel_size' in candidate_config else None

        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.channel = candidate_config[
            'channel'] if 'channel' in candidate_config else None
        self.base_channel = self._out_channels
        if self.expand_ratio != None:
            self.base_channel = int(self._out_channels / max(self.expand_ratio))

        self.transform_kernel = transform_kernel
        if self.ks_set != None:
            self.ks_set.sort()
        if self.transform_kernel != False:
            scale_param = dict()
            ### create parameter to transform kernel
            for i in range(len(self.ks_set) - 1):
                ks_small = self.ks_set[i]
                ks_large = self.ks_set[i + 1]
                param_name = '%dto%d_matrix' % (ks_large, ks_small)
                ks_t = ks_small**2
                scale_param[param_name] = self.create_parameter(
                    attr=paddle.ParamAttr(
                        name=self._full_name + param_name,
                        initializer=nn.initializer.Assign(np.eye(ks_t))),
                    shape=(ks_t, ks_t),
                    dtype=self._dtype)

            for name, param in scale_param.items():
                setattr(self, name, param)

    def get_active_filter(self, in_nc, out_nc, kernel_size):
        start, end = compute_start_end(self._kernel_size[0], kernel_size)
        ### if NOT transform kernel, intercept a center filter with kernel_size from largest filter
        filters = self.weight[:out_nc, :in_nc, start:end, start:end]
        if self.transform_kernel != False and kernel_size < self._kernel_size[
                0]:
            ### if transform kernel, then use matrix to transform
            start_filter = self.weight[:out_nc, :in_nc, :, :]
            for i in range(len(self.ks_set) - 1, 0, -1):
                src_ks = self.ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self.ks_set[i - 1]
                start, end = compute_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[(_input_filter.shape[0] * _input_filter.shape[1]),
                           -1])
                _input_filter = paddle.matmul(
                    _input_filter,
                    self.__getattr__('%dto%d_matrix' %
                                     (src_ks, target_ks)), False, False)
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[
                        filters.shape[0], filters.shape[1], target_ks, target_ks
                    ])
                start_filter = _input_filter
            filters = start_filter
        return filters

    def get_groups_in_out_nc(self, in_nc, out_nc):
        if self._groups == 1:
            ### standard conv
            return self._groups, in_nc, out_nc
        elif self._groups == self._in_channels:
            ### depthwise convolution
            if in_nc != out_nc:
                _logger.debug(
                    "input channel and output channel in depthwise conv is different, change output channel to input channel! origin channel:(in_nc {}, out_nc {}): ".
                    format(in_nc, out_nc))
            groups = in_nc
            out_nc = in_nc
            return groups, in_nc, out_nc
        else:
            ### groups convolution
            ### conv: weight: (Cout, Cin/G, Kh, Kw)
            groups = self._groups
            in_nc = int(in_nc // groups)
            return groups, in_nc, out_nc

    def forward(self, input, kernel_size=None, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): Input tensor.
            kernel_size(int, optional): the kernel size of the filter in actual calculation. Default: None.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'channel': channel
        }
        in_nc = int(input.shape[1])
        assert (
            expand_ratio == None or channel == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."
        if expand_ratio != None:
            out_nc = int(expand_ratio * self.base_channel)
        elif channel != None:
            out_nc = int(channel)
        else:
            out_nc = self._out_channels
        ks = int(self._kernel_size[0]) if kernel_size == None else int(
            kernel_size)

        groups, weight_in_nc, weight_out_nc = self.get_groups_in_out_nc(in_nc,
                                                                        out_nc)

        weight = self.get_active_filter(weight_in_nc, weight_out_nc, ks)

        if kernel_size != None or 'kernel_size' in self.candidate_config.keys():
            padding = convert_to_list(get_same_padding(ks), 2)
        else:
            padding = self._padding

        if self.bias is not None:
            ### if conv is depthwise conv, expand_ratio=0, but conv' expand 
            ### ratio before depthwise conv is not equal to 1.0, the shape of the weight
            ### about this depthwise conv is changed, but out_nc is not change,
            ### so need to change bias shape according to the weight_out_nc.
            ### if in_nc > groups > 1, the actual output of conv is weight_out_nc * groups,
            ### so slice the shape of bias by weight_out_nc and groups.
            ### if in_nc = groups, slice the shape of bias by weight_out_nc.
            if groups != in_nc:
                weight_out_nc = weight_out_nc * groups
            bias = self.bias[:weight_out_nc]
        else:
            bias = self.bias

        out = F.conv2d(
            input,
            weight,
            bias=bias,
            stride=self._stride,
            padding=padding,
            dilation=self._dilation,
            groups=groups,
            data_format=self._data_format)
        return out


class SuperGroupConv2D(SuperConv2D):
    def get_groups_in_out_nc(self, in_nc, out_nc):
        ### groups convolution
        ### conv: weight: (Cout, Cin/G, Kh, Kw)
        groups = self._groups
        in_nc = int(in_nc // groups)
        return groups, in_nc, out_nc


class SuperDepthwiseConv2D(SuperConv2D):
    ### depthwise convolution
    def get_groups_in_out_nc(self, in_nc, out_nc):
        if in_nc != out_nc:
            _logger.debug(
                "input channel and output channel in depthwise conv is different, change output channel to input channel! origin channel:(in_nc {}, out_nc {}): ".
                format(in_nc, out_nc))
        groups = in_nc
        out_nc = in_nc
        return groups, in_nc, out_nc


class SuperConv2DTranspose(nn.Conv2DTranspose):
    """
    This interface is used to construct a callable object of the ``SuperConv2DTranspose`` 
    class.

    Note: the channel in config need to less than first defined.

    The super convolution2D transpose layer calculates the output based on the input,
    filter, and dilations, strides, paddings. Input and output
    are in NCHW format. Where N is batch size, C is the number of feature map,
    H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of input feature map,
    C is the number of output feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    If bias attribution and activation type are provided, bias is added to
    the output of the convolution, and the corresponding activation function
    is applied to the final result.
    The details of convolution transpose layer, please refer to the following explanation and references
    `conv2dtranspose <http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf>`_ .
    For each input :math:`X`, the equation is:
    .. math::
        Out = \sigma (W \\ast X + b)
    Where:
    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.
    Example:
        - Input:
          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`
          Filter shape: :math:`(C_{in}, C_{out}, H_f, W_f)`
        - Output:
          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        .. math::
           H^\prime_{out} &= (H_{in} - 1) * strides[0] - 2 * paddings[0] + dilations[0] * (H_f - 1) + 1 \\\\
           W^\prime_{out} &= (W_{in} - 1) * strides[1] - 2 * paddings[1] + dilations[1] * (W_f - 1) + 1 \\\\
           H_{out} &\in [ H^\prime_{out}, H^\prime_{out} + strides[0] ) \\\\
           W_{out} &\in [ W^\prime_{out}, W^\prime_{out} + strides[1] )
    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the filter. It is as same as the output
            feature map.
        filter_size(int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        candidate_config(dict, optional): Dictionary descripts candidate config of this layer,
            such as {'kernel_size': (3, 5, 7), 'channel': (4, 6, 8)}, means the kernel size of 
            this layer can be choose from (3, 5, 7), the key of candidate_config
            only can be 'kernel_size', 'channel' and 'expand_ratio', 'channel' and 'expand_ratio'
            CANNOT be set at the same time. Default: None.
        transform_kernel(bool, optional): Whether to use transform matrix to transform a large filter
            to a small filter. Default: False.
        output_size(int or tuple, optional): The output image size. If output size is a
            tuple, it must contain two integers, (image_H, image_W). None if use
            filter_size, padding, and stride to calculate output_size.
            if output_size and filter_size are specified at the same time, They
            should follow the formula above. Default: None.
        padding(int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups(int, optional): The groups number of the Conv2d transpose layer. Inspired by
            grouped convolution in Alex Krizhevsky's Deep CNN paper, in which
            when group=2, the first half of the filters is only connected to the
            first half of the input channels, while the second half of the
            filters is only connected to the second half of the input channels.
            Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d_transpose. If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d_transpose.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d_transpose
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn(bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (Parameter): the learnable weights of filters of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    Examples:
       .. code-block:: python
          import paddle
          import numpy as np
          from paddleslim.nas.ofa.layers import SuperConv2DTranspose
          data = np.random.random((3, 32, 32, 5)).astype('float32')
          config = {'channel': 5}
          super_convtranspose = SuperConv2DTranspose(32, 10, 3)
          ret = super_convtranspose(paddle.to_tensor(data), config)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 candidate_config={},
                 transform_kernel=False,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super(SuperConv2DTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

        self.candidate_config = candidate_config
        if len(self.candidate_config.items()) != 0:
            for k, v in candidate_config.items():
                candidate_config[k] = list(set(v))
        self.ks_set = candidate_config[
            'kernel_size'] if 'kernel_size' in candidate_config else None
        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.channel = candidate_config[
            'channel'] if 'channel' in candidate_config else None
        self.base_channel = self._out_channels
        if self.expand_ratio:
            self.base_channel = int(self._out_channels / max(self.expand_ratio))

        self.transform_kernel = transform_kernel
        if self.ks_set != None:
            self.ks_set.sort()
        if self.transform_kernel != False:
            scale_param = dict()
            ### create parameter to transform kernel
            for i in range(len(self.ks_set) - 1):
                ks_small = self.ks_set[i]
                ks_large = self.ks_set[i + 1]
                param_name = '%dto%d_matrix' % (ks_large, ks_small)
                ks_t = ks_small**2
                scale_param[param_name] = self.create_parameter(
                    attr=paddle.ParamAttr(
                        name=self._full_name + param_name,
                        initializer=nn.initializer.Assign(np.eye(ks_t))),
                    shape=(ks_t, ks_t),
                    dtype=self._dtype)

            for name, param in scale_param.items():
                setattr(self, name, param)

    def get_active_filter(self, in_nc, out_nc, kernel_size):
        start, end = compute_start_end(self._kernel_size[0], kernel_size)
        filters = self.weight[:in_nc, :out_nc, start:end, start:end]
        if self.transform_kernel != False and kernel_size < self._kernel_size[
                0]:
            start_filter = self.weight[:in_nc, :out_nc, :, :]
            for i in range(len(self.ks_set) - 1, 0, -1):
                src_ks = self.ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self.ks_set[i - 1]
                start, end = compute_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[(_input_filter.shape[0] * _input_filter.shape[1]),
                           -1])
                _input_filter = paddle.matmul(
                    _input_filter,
                    self.__getattr__('%dto%d_matrix' %
                                     (src_ks, target_ks)), False, False)
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[
                        filters.shape[0], filters.shape[1], target_ks, target_ks
                    ])
                start_filter = _input_filter
            filters = start_filter
        return filters

    def get_groups_in_out_nc(self, in_nc, out_nc):
        if self._groups == 1:
            ### standard conv
            return self._groups, in_nc, out_nc
        elif self._groups == self._in_channels:
            ### depthwise convolution
            if in_nc != out_nc:
                _logger.debug(
                    "input channel and output channel in depthwise conv is different, change output channel to input channel! origin channel:(in_nc {}, out_nc {}): ".
                    format(in_nc, out_nc))
            groups = in_nc
            out_nc = in_nc
            return groups, in_nc, out_nc
        else:
            ### groups convolution
            ### groups conv transpose: weight: (Cin, Cout/G, Kh, Kw)
            groups = self._groups
            out_nc = int(out_nc // groups)
            return groups, in_nc, out_nc

    def forward(self,
                input,
                output_size=None,
                kernel_size=None,
                expand_ratio=None,
                channel=None):
        """
        Parameters:
            input(Tensor): input tensor.
            output_size(int, optional): the size of the feature map after transpose convolution. Default: None.
            kernel_size(int, optional): the kernel size of the filter in actual calculation. Default: None.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'channel': channel
        }
        in_nc = int(input.shape[1])
        assert (
            expand_ratio == None or channel == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."
        if expand_ratio != None:
            out_nc = int(expand_ratio * self.base_channel)
        elif channel != None:
            out_nc = int(channel)
        else:
            out_nc = self._out_channels

        ks = int(self._kernel_size[0]) if kernel_size == None else int(
            kernel_size)

        groups, weight_in_nc, weight_out_nc = self.get_groups_in_out_nc(in_nc,
                                                                        out_nc)

        weight = self.get_active_filter(weight_in_nc, weight_out_nc, ks)

        if kernel_size != None or 'kernel_size' in self.candidate_config.keys():
            padding = convert_to_list(get_same_padding(ks), 2)
        else:
            padding = self._padding

        if output_size is None:
            output_padding = self.output_padding
        else:
            output_padding = 0

        if self.bias is not None:
            if groups != in_nc:
                weight_out_nc = weight_out_nc * groups
            bias = self.bias[:weight_out_nc]
        else:
            bias = self.bias

        out = F.conv2d_transpose(
            input,
            weight,
            bias=bias,
            padding=padding,
            output_padding=output_padding,
            stride=self._stride,
            dilation=self._dilation,
            groups=groups,
            output_size=output_size,
            data_format=self._data_format)
        return out


class SuperGroupConv2DTranspose(SuperConv2DTranspose):
    def get_groups_in_out_nc(self, in_nc, out_nc):
        ### groups convolution
        ### groups conv transpose: weight: (Cin, Cout/G, Kh, Kw)
        groups = self._groups
        out_nc = int(out_nc // groups)
        return groups, in_nc, out_nc


class SuperDepthwiseConv2DTranspose(SuperConv2DTranspose):
    def get_groups_in_out_nc(self, in_nc, out_nc):
        if in_nc != out_nc:
            _logger.debug(
                "input channel and output channel in depthwise conv transpose is different, change output channel to input channel! origin channel:(in_nc {}, out_nc {}): ".
                format(in_nc, out_nc))
        groups = in_nc
        out_nc = in_nc
        return groups, in_nc, out_nc


### NOTE: only search channel, write for GAN-compression, maybe change to SuperDepthwiseConv and SuperConv after.
class SuperSeparableConv2D(nn.Layer):
    """
    This interface is used to construct a callable object of the ``SuperSeparableConv2D``
    class.
    The difference between ```SuperSeparableConv2D``` and ```SeparableConv2D``` is: 
    ```SuperSeparableConv2D``` need to feed a config dictionary with the format of 
    {'channel', num_of_channel} represents the channels of the first conv's outputs and
    the second conv's inputs, used to change the first dimension of weight and bias, 
    only train the first channels of the weight and bias.

    The architecture of super separable convolution2D op is [Conv2D, norm layer(may be BatchNorm2D
    or InstanceNorm2D), Conv2D]. The first conv is depthwise conv, the filter number is input channel
    multiply scale_factor, the group is equal to the number of input channel. The second conv
    is standard conv, which filter size and stride size are 1. 

    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of the second conv's filter. It is as same as the output
            feature map.
        filter_size(int or tuple): The first conv's filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        padding(int or tuple, optional): The first conv's padding size. If padding is a tuple, 
            it must contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        stride(int or tuple, optional): The first conv's stride size. If stride is a tuple,
            it must contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        dilation(int or tuple, optional): The first conv's dilation size. If dilation is a tuple, 
            it must contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        norm_layer(class): The normalization layer between two convolution. Default: InstanceNorm2D.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of convolution.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, convolution
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        scale_factor(float): The scale factor of the first conv's output channel. Default: 1.
    Returns:
        None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 candidate_config={},
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_layer=nn.InstanceNorm2D,
                 bias_attr=None,
                 scale_factor=1):
        super(SuperSeparableConv2D, self).__init__()
        self.conv = nn.LayerList([
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels * scale_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias_attr=bias_attr)
        ])

        self.conv.extend([norm_layer(in_channels * scale_factor)])

        self.conv.extend([
            nn.Conv2D(
                in_channels=in_channels * scale_factor,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=bias_attr)
        ])

        self.candidate_config = candidate_config
        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.base_output_dim = self.conv[0]._out_channels
        if self.expand_ratio != None:
            self.base_output_dim = int(self.conv[0]._out_channels /
                                       max(self.expand_ratio))

    def forward(self, input, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): input tensor.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {'expand_ratio': expand_ratio, 'channel': channel}
        in_nc = int(input.shape[1])
        assert (
            expand_ratio == None or channel == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."
        if expand_ratio != None:
            out_nc = int(expand_ratio * self.base_output_dim)
        elif channel != None:
            out_nc = int(channel)
        else:
            out_nc = self.conv[0]._out_channels

        weight = self.conv[0].weight[:in_nc]
        ###  conv1
        if self.conv[0].bias is not None:
            bias = self.conv[0].bias[:in_nc]
        else:
            bias = self.conv[0].bias

        conv0_out = F.conv2d(
            input,
            weight,
            bias,
            stride=self.conv[0]._stride,
            padding=self.conv[0]._padding,
            dilation=self.conv[0]._dilation,
            groups=in_nc,
            data_format=self.conv[0]._data_format)

        norm_out = self.conv[1](conv0_out)

        weight = self.conv[2].weight[:out_nc, :in_nc, :, :]

        if self.conv[2].bias is not None:
            bias = self.conv[2].bias[:out_nc]
        else:
            bias = self.conv[2].bias

        conv1_out = F.conv2d(
            norm_out,
            weight,
            bias,
            stride=self.conv[2]._stride,
            padding=self.conv[2]._padding,
            dilation=self.conv[2]._dilation,
            groups=self.conv[2]._groups,
            data_format=self.conv[2]._data_format)
        return conv1_out


class SuperLinear(nn.Linear):
    """
    Super Fully-connected linear transformation layer. 
    
    For each input :math:`X` , the equation is:
    .. math::
        Out = XW + b
    where :math:`W` is the weight and :math:`b` is the bias.
    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.
    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        candidate_config(dict, optional): Dictionary descripts candidate config of this layer,
            such as {'channel': (4, 6, 8)}, the key of candidate_config
            only can be 'channel' and 'expand_ratio', 'channel' and 'expand_ratio'
            CANNOT be set at the same time. Default: None.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
    Attribute:
        **weight** (Parameter): the learnable weight of this layer.
        **bias** (Parameter): the learnable bias of this layer.
    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` .
    Examples:
        .. code-block:: python
          import numpy as np
          import paddle
          from paddleslim.nas.ofa.layers import SuperLinear
          
          data = np.random.uniform(-1, 1, [32, 64]).astype('float32')
          config = {'channel': 16}
          linear = SuperLinear(64, 64)
          data = paddle.to_tensor(data)
          res = linear(data, **config)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 candidate_config={},
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(SuperLinear, self).__init__(in_features, out_features,
                                          weight_attr, bias_attr, name)
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._in_features = in_features
        self._out_features = out_features
        self.candidate_config = candidate_config
        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.base_output_dim = self._out_features
        if self.expand_ratio != None:
            self.base_output_dim = int(self._out_features /
                                       max(self.expand_ratio))

    def forward(self, input, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): input tensor.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {'expand_ratio': expand_ratio, 'channel': channel}
        ### weight: (Cin, Cout)
        in_nc = int(input.shape[-1])
        assert (
            expand_ratio == None or channel == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."
        if expand_ratio != None:
            out_nc = int(expand_ratio * self.base_output_dim)
        elif channel != None:
            out_nc = int(channel)
        else:
            out_nc = self._out_features

        weight = self.weight[:in_nc, :out_nc]
        if self._bias_attr != False:
            bias = self.bias[:out_nc]
        else:
            bias = self.bias

        out = F.linear(x=input, weight=weight, bias=bias, name=self.name)
        return out


class SuperBatchNorm2D(nn.BatchNorm2D):
    """
    This interface is used to construct a callable object of the ``SuperBatchNorm2D`` class. 

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Examples:
       .. code-block:: python
         import paddle
         import numpy as np
         from paddleslim.nas.ofa.layers import SuperBatchNorm2D
         
         np.random.seed(123)
         x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
         x = paddle.to_tensor(x_data)
         batch_norm = SuperBatchNorm2D(5)
         batch_norm_out = batch_norm(x)
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 use_global_stats=None,
                 name=None):
        super(SuperBatchNorm2D, self).__init__(
            num_features, momentum, epsilon, weight_attr, bias_attr,
            data_format, use_global_stats, name)

    def forward(self, input):
        self._check_data_format(self._data_format)
        self._check_input_dim(input)

        feature_dim = int(input.shape[1])

        weight = self.weight[:feature_dim]
        bias = self.bias[:feature_dim]
        mean = self._mean[:feature_dim]
        variance = self._variance[:feature_dim]

        return F.batch_norm(
            input,
            mean,
            variance,
            weight=weight,
            bias=bias,
            training=self.training,
            momentum=self._momentum,
            epsilon=self._epsilon,
            data_format=self._data_format,
            use_global_stats=self._use_global_stats)


class SuperSyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 name=None):
        super(SuperSyncBatchNorm,
              self).__init__(num_features, momentum, epsilon, weight_attr,
                             bias_attr, data_format, name)

    def forward(self, input):

        feature_dim = int(input.shape[1])

        weight = self.weight[:feature_dim]
        bias = self.bias[:feature_dim]
        mean = self._mean[:feature_dim]
        variance = self._variance[:feature_dim]

        mean_out = mean
        # variance and variance out share the same memory
        variance_out = variance

        attrs = ("momentum", self._momentum, "epsilon", self._epsilon,
                 "is_test", not self.training, "data_layout", self._data_format,
                 "use_mkldnn", False, "fuse_with_relu", False,
                 "use_global_stats", False, 'trainable_statistics', False)
        sync_batch_norm_out, _, _, _, _, _ = core.ops.sync_batch_norm(
            input, weight, bias, mean, variance, mean_out, variance_out, *attrs)

        return sync_batch_norm_out


class SuperInstanceNorm2D(nn.InstanceNorm2D):
    """
    This interface is used to construct a callable object of the ``SuperBatchNorm2D`` class. 

    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

    Examples:
       .. code-block:: python
         import paddle
         import numpy as np
         from paddleslim.nas.ofa.layers import SuperInstanceNorm2D
         
         np.random.seed(123)
         x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
         x = paddle.to_tensor(x_data)
         instance_norm = SuperInstanceNorm2D(5)
         out = instance_norm(x)
    """

    def __init__(self,
                 num_features,
                 epsilon=1e-05,
                 momentum=0.9,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 name=None):
        super(SuperInstanceNorm2D, self).__init__(num_features, epsilon,
                                                  momentum, weight_attr,
                                                  bias_attr, data_format, name)

    def forward(self, input):
        self._check_input_dim(input)

        feature_dim = int(input.shape[1])
        if self._weight_attr == False and self._bias_attr == False:
            scale = None
            bias = None
        else:
            scale = self.scale[:feature_dim]
            bias = self.bias[:feature_dim]

        return F.instance_norm(input, scale, bias, eps=self._epsilon)


class SuperLayerNorm(nn.LayerNorm):
    """
    This interface is used to construct a callable object of the ``SuperLayerNorm`` class.

    The difference between ```SuperLayerNorm``` and ```LayerNorm``` is: 
    the trained weight and bias in ```SuperLayerNorm``` can be changed according to the shape of input,
    only train the first channels of the weight and bias.

    Parameters:
        normalized_shape(int|list|tuple): Input shape from an expected input of
            size :math:`[*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        epsilon(float, optional): The small value added to the variance to prevent
            division by zero. Default: 1e-05.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            gain :math:`g`. If False, weight is None. If is None, a default :code:`ParamAttr` would be added as scale. The
            :attr:`param_attr` is initialized as 1 if it is added. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the learnable
            bias :math:`b`. If is False, bias is None. If is None, a default :code:`ParamAttr` would be added as bias. The
            :attr:`bias_attr` is initialized as 0 if it is added. Default: None.
        name(str, optional): Name for the LayerNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Shape:
        - x: 2-D, 3-D, 4-D or 5-D tensor.
        - output: same shape as input x.
    Returns:
        None
    Examples:
        .. code-block:: python
          import paddle
          import numpy as np
          from paddleslim.nas.ofa.layers import SuperLayerNorm
          
          np.random.seed(123)
          x_data = np.random.random(size=(2, 3)).astype('float32')
          x = paddle.to_tensor(x_data)
          layer_norm = SuperLayerNorm(x_data.shape[1])
          layer_norm_out = layer_norm(x)
    """

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(SuperLayerNorm, self).__init__(normalized_shape, epsilon,
                                             weight_attr, bias_attr, name)

    def forward(self, input):
        ### TODO(ceci3): fix if normalized_shape is not a single number
        input_ndim = len(list(input.shape))
        normalized_ndim = len(self._normalized_shape)
        begin_norm_axis = input_ndim - normalized_ndim
        feature_dim = int(input.shape[-1])
        if self._weight_attr != False:
            weight = self.weight[:feature_dim]
        else:
            weight = None
        if self._bias_attr != False:
            bias = self.bias[:feature_dim]
        else:
            bias = None
        out, _, _ = core.ops.layer_norm(input, weight, bias, 'epsilon',
                                        self._epsilon, 'begin_norm_axis',
                                        begin_norm_axis)
        return out


class SuperEmbedding(nn.Embedding):
    """
    This interface is used to construct a callable object of the ``SuperEmbedding`` class.

    Parameters:
        num_embeddings (int): Just one element which indicate the size
            of the dictionary of embeddings.
        embedding_dim:  Just one element which indicate the size of each embedding vector respectively.
        padding_idx(int|long|None): padding_idx needs to be in the interval [-num_embeddings, num_embeddings).
            If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted
            to :math:`vocab\_size + padding\_idx` . It will output all-zero padding data whenever lookup
            encounters :math:`padding\_idx` in id. And the padding data will not be updated while training.
            If set None, it makes no effect to output. Default: None.
        sparse(bool): The flag indicating whether to use sparse update. This parameter only
            affects the performance of the backwards gradient update. It is recommended to set
            True because sparse update is faster. But some optimizer does not support sparse update,
            such as :ref:`api_optimizer_AdadeltaOptimizer` , :ref:`api_optimizer_AdamaxOptimizer` ,
            :ref:`api_optimizer_DecayedAdagradOptimizer` , :ref:`api_optimizer_FtrlOptimizer` ,
            :ref:`api_optimizer_LambOptimizer` and :ref:`api_optimizer_LarsMomentumOptimizer` .
            In these case, sparse must be False. Default: False.
        weight_attr(ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_ParamAttr` . In addition,
            user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter.
            The local word vector needs to be transformed into numpy format, and the shape of local word
            vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_initializer_NumpyArrayInitializer`
            is used to load custom or pre-trained word vectors. See code example for details.
        name(str|None): For detailed information, please refer
               to :ref:`api_guide_Name`. Usually name is no need to set and
               None by default.
    Attribute:
        **weight** (Parameter): the learnable weights of this layer.
    Returns:
        None
    Examples:
        .. code-block:: python
          import numpy as np
          import paddle
          from paddleslim.nas.ofa.layers import SuperEmbedding
          
          data = np.random.uniform(-1, 1, [32, 64]).astype('int64')
          config = {'channel': 16}
          emb = SuperEmbedding(64, 64)
          data = paddle.to_tensor(data)
          res = emb(data, **config)
    """

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 candidate_config={},
                 padding_idx=None,
                 sparse=False,
                 weight_attr=None,
                 name=None):
        super(SuperEmbedding, self).__init__(num_embeddings, embedding_dim,
                                             padding_idx, sparse, weight_attr,
                                             name)
        self.candidate_config = candidate_config
        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.base_output_dim = self._embedding_dim
        if self.expand_ratio != None:
            self.base_output_dim = int(self._embedding_dim /
                                       max(self.expand_ratio))

    def forward(self, input, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): input tensor.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        assert (
            expand_ratio == None or channel == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."
        if expand_ratio != None:
            out_nc = int(expand_ratio * self.base_output_dim)
        elif channel != None:
            out_nc = int(channel)
        else:
            out_nc = self._embedding_dim

        weight = self.weight[:, :out_nc]
        return F.embedding(
            input,
            weight=weight,
            padding_idx=self._padding_idx,
            sparse=self._sparse,
            name=self._name)
