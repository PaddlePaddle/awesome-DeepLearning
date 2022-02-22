#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import inspect
import decorator
import logging
import numbers
import paddle
from ...common import get_logger
from .utils.utils import get_paddle_version
pd_ver = get_paddle_version()
if pd_ver == 185:
    import paddle.fluid.dygraph.nn as nn
    from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, Linear, LayerNorm, Embedding
    from paddle.fluid import ParamAttr
    from .layers_old import *
    from . import layers_old as layers
    Layer = paddle.fluid.dygraph.Layer
else:
    import paddle.nn as nn
    from paddle.nn import Conv2D, Conv2DTranspose, Linear, LayerNorm, Embedding, SyncBatchNorm
    from paddle import ParamAttr
    from .layers import *
    from . import layers
    Layer = paddle.nn.Layer
from .layers_base import Block

_logger = get_logger(__name__, level=logging.INFO)

__all__ = ['supernet', 'Convert']

WEIGHT_LAYER = ['conv', 'linear', 'embedding']


class Convert:
    """
    Convert network to the supernet according to the search space.
    Parameters:
        context(paddleslim.nas.ofa.supernet): search space defined by the user.
    Examples:
        .. code-block:: python
          from paddleslim.nas.ofa import supernet, Convert
          sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
          convert = Convert(sp_net_config)
    """

    def __init__(self, context):
        self.context = context

    def _change_name(self, layer, pd_ver, has_bias=True, conv=False):
        if conv:
            w_attr = layer._param_attr
        else:
            w_attr = layer._param_attr if pd_ver == 185 else layer._weight_attr

        if isinstance(w_attr, ParamAttr):
            if w_attr != None and not isinstance(w_attr,
                                                 bool) and w_attr.name != None:
                w_attr.name = 'super_' + w_attr.name

        if has_bias:
            if isinstance(layer._bias_attr, ParamAttr):
                if layer._bias_attr != None and not isinstance(
                        layer._bias_attr,
                        bool) and layer._bias_attr.name != None:
                    layer._bias_attr.name = 'super_' + layer._bias_attr.name

    def convert(self, network):
        """
        The function to convert the network to a supernet.
        Parameters:
            network(paddle.nn.Layer|list(paddle.nn.Layer)): instance of the model or list of instance of layers.
        Examples:
            .. code-block:: python
              from paddle.vision.models import mobilenet_v1
              from paddleslim.nas.ofa import supernet, Convert
              sp_net_config = supernet(kernel_size=(3, 5, 7), expand_ratio=[1, 2, 4])
              convert = Convert(sp_net_config).convert(mobilenet_v1())
        """
        # search the first and last weight layer, don't change out channel of the last weight layer
        # don't change in channel of the first weight layer
        model = []
        if isinstance(network, Layer):
            for name, sublayer in network.named_sublayers():
                model.append(sublayer)
        else:
            model = network

        first_weight_layer_idx = -1
        last_weight_layer_idx = -1
        weight_layer_count = 0
        # NOTE: pre_channel store for shortcut module
        pre_channel = None
        cur_channel = None
        for idx, layer in enumerate(model):
            cls_name = layer.__class__.__name__.lower()
            ### basic api in paddle
            if len(layer.sublayers()) == 0:
                if 'conv' in cls_name or 'linear' in cls_name or 'embedding' in cls_name:
                    weight_layer_count += 1
                    last_weight_layer_idx = idx
                    if first_weight_layer_idx == -1:
                        first_weight_layer_idx = idx

        if getattr(self.context, 'channel', None) != None:
            assert len(
                self.context.channel
            ) == weight_layer_count, "length of channel must same as weight layer."

        for idx, layer in enumerate(model):
            if isinstance(layer, Conv2D):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    'stride', 'padding', 'dilation', 'groups', 'bias_attr'
                ]
                if pd_ver == 185:
                    new_attr_name += ['param_attr', 'use_cudnn', 'act', 'dtype']
                else:
                    new_attr_name += [
                        'weight_attr', 'data_format', 'padding_mode'
                    ]

                self._change_name(layer, pd_ver, conv=True)
                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['candidate_config'] = dict()
                if pd_ver == 185:
                    new_attr_dict['num_channels'] = None
                    new_attr_dict['num_filters'] = None
                    new_attr_dict['filter_size'] = None
                else:
                    new_attr_dict['in_channels'] = None
                    new_attr_dict['out_channels'] = None
                    new_attr_dict['kernel_size'] = None
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                # if the kernel_size of conv is 1, don't change it.
                fks = '_filter_size' if '_filter_size' in attr_dict.keys(
                ) else '_kernel_size'

                ks = [attr_dict[fks]] if isinstance(
                    attr_dict[fks], numbers.Integral) else attr_dict[fks]

                if self.kernel_size and int(ks[0]) != 1:
                    new_attr_dict['transform_kernel'] = True
                    new_attr_dict[fks[1:]] = max(self.kernel_size)
                    new_attr_dict['candidate_config'].update({
                        'kernel_size': self.kernel_size
                    })
                else:
                    new_attr_dict[fks[1:]] = attr_dict[fks]

                in_key = '_num_channels' if '_num_channels' in attr_dict.keys(
                ) else '_in_channels'
                out_key = '_num_filters' if '_num_filters' in attr_dict.keys(
                ) else '_out_channels'
                if self.context.expand:
                    ### first super convolution
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    else:
                        new_attr_dict[in_key[1:]] = int(self.context.expand *
                                                        attr_dict[in_key])

                    ### last super convolution
                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = attr_dict[out_key]
                    else:
                        new_attr_dict[out_key[1:]] = int(self.context.expand *
                                                         attr_dict[out_key])
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    if attr_dict['_groups'] != None and (
                            int(attr_dict['_groups']) == int(attr_dict[in_key])
                    ):
                        ### depthwise conv, if conv is depthwise, use pre channel as cur_channel
                        _logger.warn(
                        "If convolution is a depthwise conv, output channel change" \
                        " to the same channel with input, output channel in search is not used."
                        )
                        cur_channel = pre_channel
                    else:
                        cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    else:
                        new_attr_dict[in_key[1:]] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = attr_dict[out_key]
                    else:
                        new_attr_dict[out_key[1:]] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    new_attr_dict[out_key[1:]] = attr_dict[out_key]

                for attr in new_attr_name:
                    if attr == 'weight_attr':
                        new_attr_dict[attr] = attr_dict['_param_attr']
                    else:
                        new_attr_dict[attr] = attr_dict['_' + attr]

                del layer

                if attr_dict['_groups'] == None or int(attr_dict[
                        '_groups']) == 1:
                    ### standard conv
                    layer = Block(SuperConv2D(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[in_key]):
                    # if conv is depthwise conv, groups = in_channel, out_channel = in_channel,
                    # channel in candidate_config = in_channel_list
                    if 'channel' in new_attr_dict['candidate_config']:
                        new_attr_dict[in_key[1:]] = max(cur_channel)
                        new_attr_dict[out_key[1:]] = new_attr_dict[in_key[1:]]
                        new_attr_dict['candidate_config'][
                            'channel'] = cur_channel
                    new_attr_dict['groups'] = new_attr_dict[in_key[1:]]
                    layer = Block(
                        SuperDepthwiseConv2D(**new_attr_dict), key=key)
                else:
                    ### group conv
                    layer = Block(SuperGroupConv2D(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer,
                            getattr(nn, 'BatchNorm2D', nn.BatchNorm)) and (
                                getattr(self.context, 'expand', None) != None or
                                getattr(self.context, 'channel', None) != None):
                # num_features in BatchNorm don't change after last weight operators
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = ['momentum', 'epsilon', 'bias_attr']

                if pd_ver == 185:
                    new_attr_name += [
                        'param_attr', 'act', 'dtype', 'in_place', 'data_layout',
                        'is_test', 'use_global_stats', 'trainable_statistics'
                    ]
                else:
                    new_attr_name += ['weight_attr', 'data_format', 'name']

                self._change_name(layer, pd_ver)
                new_attr_dict = dict.fromkeys(new_attr_name, None)
                if pd_ver == 185:
                    new_attr_dict['num_channels'] = None
                else:
                    new_attr_dict['num_features'] = None
                new_key = 'num_channels' if 'num_channels' in new_attr_dict.keys(
                ) else 'num_features'
                if self.context.expand:
                    new_attr_dict[new_key] = int(
                        self.context.expand *
                        layer._parameters['weight'].shape[0])
                elif self.context.channel:
                    new_attr_dict[new_key] = max(cur_channel)
                else:
                    new_attr_dict[new_key] = attr_dict[
                        '_num_channels'] if '_num_channels' in attr_dict.keys(
                        ) else attr_dict['_num_features']

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                del layer, attr_dict

                layer = layers.SuperBatchNorm(
                    **new_attr_dict
                ) if pd_ver == 185 else layers.SuperBatchNorm2D(**new_attr_dict)
                model[idx] = layer

            elif isinstance(layer, SyncBatchNorm) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                # num_features in SyncBatchNorm don't change after last weight operators
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = ['momentum', 'epsilon', 'bias_attr']

                new_attr_name += ['weight_attr', 'data_format', 'name']

                self._change_name(layer, pd_ver)
                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['num_features'] = None

                new_key = 'num_channels' if 'num_channels' in new_attr_dict.keys(
                ) else 'num_features'

                if self.context.expand:
                    new_attr_dict[new_key] = int(
                        self.context.expand *
                        layer._parameters['weight'].shape[0])
                elif self.context.channel:
                    new_attr_dict[new_key] = max(cur_channel)
                else:
                    new_attr_dict[new_key] = attr_dict[
                        '_num_channels'] if '_num_channels' in attr_dict.keys(
                        ) else attr_dict['_num_features']

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                del layer, attr_dict

                layer = layers.SuperSyncBatchNorm(**new_attr_dict)
                model[idx] = layer

            ### assume output_size = None, filter_size != None
            ### NOTE: output_size != None may raise error, solve when it happend. 
            elif isinstance(layer, Conv2DTranspose):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']

                new_attr_name = [
                    'stride', 'padding', 'dilation', 'groups', 'bias_attr'
                ]
                assert getattr(
                    attr_dict, '_filter_size', '_kernel_size'
                ) != None, "Conv2DTranspose only support kernel size != None now"

                if pd_ver == 185:
                    new_attr_name += [
                        'output_size', 'param_attr', 'use_cudnn', 'act', 'dtype'
                    ]
                else:
                    new_attr_name += [
                        'output_padding', 'weight_attr', 'data_format'
                    ]

                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['candidate_config'] = dict()
                if pd_ver == 185:
                    new_attr_dict['num_channels'] = None
                    new_attr_dict['num_filters'] = None
                    new_attr_dict['filter_size'] = None
                else:
                    new_attr_dict['in_channels'] = None
                    new_attr_dict['out_channels'] = None
                    new_attr_dict['kernel_size'] = None

                self._change_name(layer, pd_ver, conv=True)
                self.kernel_size = getattr(self.context, 'kernel_size', None)

                # if the kernel_size of conv transpose is 1, don't change it.
                fks = '_filter_size' if '_filter_size' in attr_dict.keys(
                ) else '_kernel_size'
                ks = [attr_dict[fks]] if isinstance(
                    attr_dict[fks], numbers.Integral) else attr_dict[fks]

                if self.kernel_size and int(ks[0]) != 1:
                    new_attr_dict['transform_kernel'] = True
                    new_attr_dict[fks[1:]] = max(self.kernel_size)
                    new_attr_dict['candidate_config'].update({
                        'kernel_size': self.kernel_size
                    })
                else:
                    new_attr_dict[fks[1:]] = attr_dict[fks]

                in_key = '_num_channels' if '_num_channels' in attr_dict.keys(
                ) else '_in_channels'
                out_key = '_num_filters' if '_num_filters' in attr_dict.keys(
                ) else '_out_channels'
                if self.context.expand:
                    ### first super convolution transpose
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    else:
                        new_attr_dict[in_key[1:]] = int(self.context.expand *
                                                        attr_dict[in_key])
                    ### last super convolution transpose
                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = attr_dict[out_key]
                    else:
                        new_attr_dict[out_key[1:]] = int(self.context.expand *
                                                         attr_dict[out_key])
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    if attr_dict['_groups'] != None and (
                            int(attr_dict['_groups']) == int(attr_dict[in_key])
                    ):
                        ### depthwise conv_transpose
                        _logger.warn(
                        "If convolution is a depthwise conv_transpose, output channel " \
                        "change to the same channel with input, output channel in search is not used."
                        )
                        cur_channel = pre_channel
                    else:
                        cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    else:
                        new_attr_dict[in_key[1:]] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = attr_dict[out_key]
                    else:
                        new_attr_dict[out_key[1:]] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict[in_key[1:]] = attr_dict[in_key]
                    new_attr_dict[out_key[1:]] = attr_dict[out_key]

                for attr in new_attr_name:
                    if attr == 'weight_attr':
                        new_attr_dict[attr] = attr_dict['_param_attr']
                    elif attr == 'output_padding':
                        new_attr_dict[attr] = attr_dict[attr]
                    else:
                        new_attr_dict[attr] = attr_dict['_' + attr]

                del layer

                if getattr(new_attr_dict, 'output_size', None) == []:
                    new_attr_dict['output_size'] = None

                if attr_dict['_groups'] == None or int(attr_dict[
                        '_groups']) == 1:
                    ### standard conv_transpose
                    layer = Block(
                        SuperConv2DTranspose(**new_attr_dict), key=key)
                elif int(attr_dict['_groups']) == int(attr_dict[in_key]):
                    # if conv is depthwise conv, groups = in_channel, out_channel = in_channel,
                    # channel in candidate_config = in_channel_list
                    if 'channel' in new_attr_dict['candidate_config']:
                        new_attr_dict[in_key[1:]] = max(cur_channel)
                        new_attr_dict[out_key[1:]] = new_attr_dict[in_key[1:]]
                        new_attr_dict['candidate_config'][
                            'channel'] = cur_channel
                    new_attr_dict['groups'] = new_attr_dict[in_key[1:]]
                    layer = Block(
                        SuperDepthwiseConv2DTranspose(**new_attr_dict), key=key)
                else:
                    ### group conv_transpose
                    layer = Block(
                        SuperGroupConv2DTranspose(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(layer, Linear) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                if pd_ver == 185:
                    new_attr_name = ['act', 'dtype']
                else:
                    new_attr_name = ['weight_attr', 'bias_attr']
                self._change_name(layer, pd_ver) if pd_ver != 185 else None
                in_nc, out_nc = layer._parameters['weight'].shape

                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['candidate_config'] = dict()
                if pd_ver == 185:
                    new_attr_dict['input_dim'] = None
                    new_attr_dict['output_dim'] = None
                else:
                    new_attr_dict['in_features'] = None
                    new_attr_dict['out_features'] = None

                in_key = '_input_dim' if pd_ver == 185 else '_in_features'
                out_key = '_output_dim' if pd_ver == 185 else '_out_features'
                attr_dict[in_key] = in_nc
                attr_dict[out_key] = out_nc
                if self.context.expand:
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = int(attr_dict[in_key])
                    else:
                        new_attr_dict[in_key[1:]] = int(self.context.expand *
                                                        attr_dict[in_key])

                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = int(attr_dict[out_key])
                    else:
                        new_attr_dict[out_key[1:]] = int(self.context.expand *
                                                         attr_dict[out_key])
                        new_attr_dict['candidate_config'].update({
                            'expand_ratio': self.context.expand_ratio
                        })
                elif self.context.channel:
                    cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if idx == first_weight_layer_idx:
                        new_attr_dict[in_key[1:]] = int(attr_dict[in_key])
                    else:
                        new_attr_dict[in_key[1:]] = max(pre_channel)

                    if idx == last_weight_layer_idx:
                        new_attr_dict[out_key[1:]] = int(attr_dict[out_key])
                    else:
                        new_attr_dict[out_key[1:]] = max(cur_channel)
                        new_attr_dict['candidate_config'].update({
                            'channel': cur_channel
                        })
                        pre_channel = cur_channel
                else:
                    new_attr_dict[in_key[1:]] = int(attr_dict[in_key])
                    new_attr_dict[out_key[1:]] = int(attr_dict[out_key])

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                del layer, attr_dict

                layer = Block(SuperLinear(**new_attr_dict), key=key)
                model[idx] = layer

            elif isinstance(
                    layer,
                    getattr(nn, 'InstanceNorm2D',
                            paddle.fluid.dygraph.nn.InstanceNorm)) and (
                                getattr(self.context, 'expand', None) != None or
                                getattr(self.context, 'channel', None) != None):
                # num_features in InstanceNorm don't change after last weight operators
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                if pd_ver == 185:
                    new_attr_name = [
                        'bias_attr', 'epsilon', 'param_attr', 'dtype'
                    ]
                else:
                    new_attr_name = ['bias_attr', 'epsilon', 'weight_attr']

                self._change_name(layer, pd_ver)
                new_attr_dict = dict.fromkeys(new_attr_name, None)
                if pd_ver == 185:
                    new_attr_dict['num_channels'] = None
                else:
                    new_attr_dict['num_features'] = None
                new_key = '_num_channels' if '_num_channels' in new_attr_dict.keys(
                ) else '_num_features'
                ### 10 is a default channel in the case of weight_attr=False, in this condition, num of channels if useless, so give it arbitrarily.
                attr_dict[new_key] = layer._parameters['scale'].shape[0] if len(
                    layer._parameters) != 0 else 10

                if self.context.expand:
                    new_attr_dict[new_key[1:]] = int(self.context.expand *
                                                     attr_dict[new_key])
                elif self.context.channel:
                    new_attr_dict[new_key[1:]] = max(cur_channel)
                else:
                    new_attr_dict[new_key[1:]] = attr_dict[new_key]

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                del layer, attr_dict

                layer = layers.SuperInstanceNorm(
                    **new_attr_dict
                ) if pd_ver == 185 else layers.SuperInstanceNorm2D(
                    **new_attr_dict)
                model[idx] = layer

            elif isinstance(layer, LayerNorm) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                ### TODO(ceci3): fix when normalized_shape != last_dim_of_input
                if idx > last_weight_layer_idx:
                    continue

                attr_dict = layer.__dict__
                new_attr_name = ['epsilon', 'bias_attr']
                if pd_ver == 185:
                    new_attr_name += [
                        'scale', 'shift', 'param_attr', 'act', 'dtype'
                    ]
                else:
                    new_attr_name += ['weight_attr']

                self._change_name(layer, pd_ver)
                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['normalized_shape'] = None
                if self.context.expand:
                    new_attr_dict['normalized_shape'] = int(
                        self.context.expand * attr_dict['_normalized_shape'][0])
                elif self.context.channel:
                    new_attr_dict['normalized_shape'] = max(cur_channel)
                else:
                    new_attr_dict['normalized_shape'] = attr_dict[
                        '_normalized_shape']

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                del layer, attr_dict
                layer = SuperLayerNorm(**new_attr_dict)
                model[idx] = layer

            elif isinstance(layer, Embedding) and (
                    getattr(self.context, 'expand', None) != None or
                    getattr(self.context, 'channel', None) != None):
                attr_dict = layer.__dict__
                key = attr_dict['_full_name']
                new_attr_name = []
                if pd_ver == 185:
                    new_attr_name += [
                        'is_sparse', 'is_distributed', 'param_attr', 'dtype'
                    ]
                else:
                    new_attr_name += ['sparse', 'weight_attr', 'name']

                self._change_name(layer, pd_ver, has_bias=False)

                new_attr_dict = dict.fromkeys(new_attr_name, None)
                new_attr_dict['candidate_config'] = dict()
                bef_size = attr_dict['_size']
                if self.context.expand:
                    if pd_ver == 185:
                        new_attr_dict['size'] = [
                            bef_size[0], int(self.context.expand * bef_size[1])
                        ]
                    else:
                        new_attr_dict['num_embeddings'] = attr_dict[
                            '_num_embeddings']
                        new_attr_dict['embedding_dim'] = int(
                            self.context.expand * attr_dict['_embedding_dim'])

                    new_attr_dict['candidate_config'].update({
                        'expand_ratio': self.context.expand_ratio
                    })

                elif self.context.channel:
                    cur_channel = self.context.channel[0]
                    self.context.channel = self.context.channel[1:]
                    if pd_ver == 185:
                        new_attr_dict['size'] = [bef_size[0], max(cur_channel)]
                    else:
                        new_attr_dict['num_embeddings'] = attr_dict[
                            '_num_embeddings']
                        new_attr_dict['embedding_dim'] = max(cur_channel)

                    new_attr_dict['candidate_config'].update({
                        'channel': cur_channel
                    })
                    pre_channel = cur_channel
                else:
                    if pf_ver == 185:
                        new_attr_dict['size'] = bef_size
                    else:
                        new_attr_dict['num_embeddings'] = attr_dict[
                            '_num_embeddings']
                        new_attr_dict['embedding_dim'] = attr_dict[
                            '_embedding_dim']

                for attr in new_attr_name:
                    new_attr_dict[attr] = attr_dict['_' + attr]

                new_attr_dict['padding_idx'] = None if attr_dict[
                    '_padding_idx'] == -1 else attr_dict['_padding_idx']

                del layer, attr_dict

                layer = Block(SuperEmbedding(**new_attr_dict), key=key)
                model[idx] = layer

        def split_prefix(net, name_list):
            if len(name_list) > 1:
                net = split_prefix(getattr(net, name_list[0]), name_list[1:])
            elif len(name_list) == 1:
                net = getattr(net, name_list[0])
            else:
                raise NotImplementedError("name error")
            return net

        def get_split_names(layer, name_list):
            if name_list:
                self.name_list.append(name_list)
            for _, (name, sublayer) in enumerate(layer.named_children()):
                if sublayer.named_children():
                    get_split_names(sublayer, name_list + [name])

        if isinstance(network, Layer):
            curr_id = 0
            self.name_list = []
            get_split_names(network, [])
            for idx, nl in enumerate(self.name_list):
                if len(nl) > 1:
                    net = split_prefix(network, nl[:-1])
                else:
                    net = network
                setattr(net, nl[-1], model[idx])

        return network


class supernet:
    """
    Search space of the network.
    Parameters:
        kernel_size(list|tuple, optional): search space for the kernel size of the Conv2D.
        expand_ratio(list|tuple, optional): the search space for the expand ratio of the number of channels of Conv2D, the expand ratio of the output dimensions of the Embedding or Linear, which means this parameter get the number of channels of each OP in the converted super network based on the the channels of each OP in the original model, so this parameter The length is 1. Just set one between this parameter and ``channel``.
        channel(list|tuple, optional): the search space for the number of channels of Conv2D, the output dimensions of the Embedding or Linear, this parameter directly sets the number of channels of each OP in the super network, so the length of this parameter needs to be the same as the total number that of Conv2D, Embedding, and Linear included in the network. Just set one between this parameter and ``expand_ratio``.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert (
            getattr(self, 'expand_ratio', None) == None or
            getattr(self, 'channel', None) == None
        ), "expand_ratio and channel CANNOT be NOT None at the same time."

        self.expand = None
        if 'expand_ratio' in kwargs.keys():
            if isinstance(self.expand_ratio, list) or isinstance(
                    self.expand_ratio, tuple):
                self.expand = max(self.expand_ratio)
            elif isinstance(self.expand_ratio, int):
                self.expand = self.expand_ratio
        if 'channel' not in kwargs.keys():
            self.channel = None

    def __enter__(self):
        return Convert(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.expand = None
        self.channel = None
        self.kernel_size = None


#def ofa_supernet(kernel_size, expand_ratio):
#    def _ofa_supernet(func):
#        @functools.wraps(func)
#        def convert(*args, **kwargs):
#            supernet_convert(*args, **kwargs)
#        return convert
#    return _ofa_supernet
