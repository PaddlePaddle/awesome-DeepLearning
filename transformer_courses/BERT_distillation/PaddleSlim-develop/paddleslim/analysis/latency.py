"""Define latency evaluators that evaluate the performance of mode on devices.
"""
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

from paddle.fluid import Program
from ..core import GraphWrapper, OpWrapper
__all__ = ["LatencyEvaluator", "TableLatencyEvaluator"]


class LatencyEvaluator(object):
    """Base class of latency evaluator.
    """

    def latency(self, graph):
        """Get latency of graph. It is an abstract method.

        Args:
            graph(GrapWrapper | Program): The graph to be evaluated.

        Returns:
            latency(float): The latency of given graph on current evaluator.
        """
        raise NotImplementedError('Abstract method.')

    def _get_ops_from_graph(self, graph, only_conv):
        assert isinstance(graph, GraphWrapper)
        ops = []
        i = 0
        for op in graph.ops():
            if op.type() in ['conv2d', 'depthwise_conv2d']:
                tmp = self._conv_op_args(op)
            elif op.type() in [
                    'elementwise_add', 'elementwise_mul', 'elementwise_max'
            ] and only_conv == False:
                tmp = self._eltwise_op_args(op)
            elif op.type() in [
                    'relu', 'prelu', 'sigmoid', 'relu6', 'elu', 'brelu',
                    'leaky_relu'
            ] and only_conv == False:
                tmp = self._activation_op_args(op)
            elif op.type() == 'batch_norm' and only_conv == False:
                tmp = self._batch_norm_op_args(op)
            elif op.type() == 'pool2d' and only_conv == False:
                tmp = self._pooling_op_args(op)
            elif op.type() == 'softmax' and only_conv == False:
                tmp = self._softmax_op_args(op)
            elif op.type() == 'mul' and only_conv == False:
                tmp = self._fc_op_args(op)
            else:
                tmp = None
            if tmp:
                ops.append(tmp)
        return ops

    def _conv_op_args(self, op):
        assert isinstance(op, OpWrapper)
        tmp, res = [], []
        # op_name
        tmp.append('conv')
        # flag_bias
        if len(op.inputs('Bias')) == 0:
            tmp.append(0)
        else:
            tmp.append(1)
        # flag_relu
        tmp.append(int(op.attr('fuse_relu')))
        # batch size
        tmp.append(1)
        # channels, height, width
        in_shapes = op.inputs('Input')[0].shape()
        tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]

        # output channels
        w_shapes = op.inputs('Filter')[0].shape()
        tmp.append(int(w_shapes[0]))

        # group
        tmp.append(int(op.attr('groups')))

        # kernel size
        tmp.append(int(w_shapes[2]))
        if w_shapes[2] != w_shapes[3]:
            res.append(int(w_shapes[3]))

        # padding
        paddings = op.attr('paddings')
        tmp.append(int(paddings[0]))
        if paddings[0] != paddings[1]:
            res.append(int(paddings[0]))

        # strides
        strides = op.attr('strides')
        tmp.append(int(strides[0]))
        if strides[0] != strides[1]:
            res.append(int(strides[1]))

        # dilations
        dilations = op.attr('dilations')
        tmp.append(int(dilations[0]))
        if dilations[0] != dilations[1]:
            res.append(int(dilations[1]))
        tmp = tmp + res
        return tmp

    def _batch_norm_op_args(self, op):
        tmp = []
        # op name
        tmp.append('batch_norm')
        # activation type
        if not op.attr('fuse_with_relu'):
            tmp.append('None')
        else:
            tmp.append('relu')
        # batch size
        tmp.append(1)
        # input channels, height, width
        in_shapes = op.inputs("X")[0].shape()
        tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]
        return tmp

    def _eltwise_op_args(self, op):
        # op name
        tmp = ['eltwise']
        # elementwise type, TODO: add more ops
        if op.type() == 'elementwise_mul':
            tmp.append(1)
        elif op.type() == 'elementwise_add':
            tmp.append(2)
        else:
            tmp.append(3)
        # batch size
        tmp.append(1)
        # input channels, height, width 
        in_shapes = op.inputs('X')[0].shape()
        while len(in_shapes) < 4:
            in_shapes = in_shapes + (1, )

        for i in range(1, len(in_shapes)):
            tmp.append(int(in_shapes[i]))
        return tmp

    def _activation_op_args(self, op):
        tmp = []
        # activation type
        tmp.append(op.type())
        # batch size
        tmp.append(1)
        # input channels, height, width
        in_shapes = op.inputs('X')[0].shape()
        while len(in_shapes) < 4:
            in_shapes = in_shapes + (1, )

        for i in range(1, len(in_shapes)):
            tmp.append(int(in_shapes[i]))
        return tmp

    def _pooling_op_args(self, op):
        tmp, res = [], []
        # op name
        tmp.append('pooling')
        # global pooling
        tmp.append(int(op.attr('global_pooling')))
        # batch size
        tmp.append(1)
        # channels, height, width
        in_shapes = op.inputs('X')[0].shape()
        tmp = tmp + [int(in_shapes[1]), int(in_shapes[2]), int(in_shapes[3])]
        # kernel size
        ksize = op.attr('ksize')
        tmp.append(int(ksize[0]))
        if ksize[0] != ksize[1]:
            res.append(int(ksize[1]))

        # padding
        paddings = op.attr('paddings')
        tmp.append(int(paddings[0]))
        if paddings[0] != paddings[1]:
            res.append(int(paddings[1]))

        # stride
        strides = op.attr('strides')
        tmp.append(int(strides[0]))
        if strides[0] != strides[1]:
            res.append(int(strides[1]))

        # ceil mode
        tmp.append(int(op.attr('ceil_mode')))

        # pool type
        pool_type = op.attr('pooling_type')
        exclusive = op.attr('exclusive')
        if pool_type == 'max' and (not exclusive):
            tmp.append(1)
        elif pool_type == 'avg' and (not exclusive):
            tmp.append(2)
        else:
            tmp.append(3)

        tmp = tmp + res
        return tmp

    def _softmax_op_args(self, op):
        # op name
        tmp = ['softmax']
        # axis
        tmp.append(op.attr('axis'))
        # batch size
        tmp.append(1)
        # input channels, height, width
        in_shapes = op.inputs('X')[0].shape()
        while len(in_shapes) < 4:
            in_shapes = in_shapes + (1, )

        for i in range(1, len(in_shapes)):
            tmp.append(int(in_shapes[i]))

        return tmp

    def _fc_op_args(self, op):
        # op name
        tmp = ['conv']
        # flag bias
        tmp.append(0)
        # flag relu
        tmp.append(0)
        # batch size 
        tmp.append(1)
        # input channels, height, width
        channels = 1
        in_shape = op.inputs('X')[0].shape()
        for i in range(1, len(in_shape)):
            channels *= in_shape[i]
        tmp = tmp + [int(channels), 1, 1]
        # output channels
        tmp.append(int(op.outputs('Out')[0].shape()[1]))
        # groups, kernel size, padding, stride, dilation
        tmp = tmp + [1, 1, 0, 1, 1]
        return tmp


class TableLatencyEvaluator(LatencyEvaluator):
    """The evaluator used to get graph's latency on some devices and infer engines.

    Args:
      table_file(str): The path of file that records the devices latency of operators.
      delimiter(str): The delimiter used in `table_file`.
    """

    def __init__(self, table_file, delimiter=","):
        self._table = self._load_table(table_file)
        self._delimiter = delimiter

    def _load_table(self, table_file):
        table = {}
        with open(table_file) as f:
            line = f.readline()
            self.infer_engine_name, self.device_name, self.create_time = line.strip(
            ).split("\t")
            for line in f:
                op_str, latency = line.strip().split("\t")
                table[op_str] = float(latency)
        return table

    def _op_latency(self, op_str):
        assert op_str in self._table
        return self._table[op_str]

    def latency(self, graph, only_conv=True):
        """Get latency of target graph.

        Args:
            graph(GrapWrapper | Program): The graph to be evaluated.
            only_conv(bool): only evaluated convolution layer if `only_conv` is true. Default: True.

        Returns:
            latency(float): The latency of given graph on current evaluator.
        """
        total_latency = 0
        if isinstance(graph, Program):
            graph = GraphWrapper(graph)
        assert isinstance(graph, GraphWrapper)
        for op in self._get_ops_from_graph(graph, only_conv):
            total_latency += self._op_latency(
                self._delimiter.join(map(lambda x: str(x), op)))
        return total_latency
