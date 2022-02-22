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
import paddle
import numpy as np
from ..core import GraphWrapper, dygraph2program

__all__ = ["flops", "dygraph_flops"]


def flops(model, inputs=None, dtypes=None, only_conv=True, detail=False):
    """
    Compute the FLOPs of nn.Layer of paddle.Program.
    Args:
      model(paddle.nn.Layer|paddle.static.Program): The target model.
      inputs(list): It is only used when model is instance of 'paddle.nn.Layer'. The dummy inputs used for 'model.forward'. It can be:
                      1. list<int>|tuple<int>: means 'model.forward' accepts
                         only one variable as argument and the shape of
                         variable is 'inputs'.
                      2. list<list<list>>: means 'model.forward' accepts multiple
                         variables as arguments and the shapes of variables is 'inputs'.
                      3. others: 'inputs' will be used as argument list by calling
                         'model.forward(*inputs)'.
      dtypes(str|list<str>): It only used when 'inputs' is shape or shapes that means
                      data type of each input. None means all the inputs is 'float32'.
                      Default: None.
      only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
      detail(bool): Whether to return detail of each convolution layer.
    """
    if isinstance(model, paddle.static.Program):
        return _static_flops(model, only_conv=only_conv, detail=detail)
    elif isinstance(model, paddle.nn.Layer):
        return dygraph_flops(
            model, inputs, dtypes=dtypes, only_conv=only_conv, detail=detail)


def _static_flops(program, only_conv=True, detail=False):
    """Get FLOPs of target graph.

    Args:
        program(Program): The program used to calculate FLOPS.
        only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
        detail(bool): Whether to return detail of each convolution layer.
    
    Returns:
        int|tuple: If `detail` is true, then return a tuple in format `(FLOPs, details)`, otherwise it will just return `FlOPs`. The details is a dict whose key is the parameter name of convlution layer and value is the FLOPs of each convolution layer. 
    """
    graph = GraphWrapper(program)
    return _graph_flops(graph, only_conv=only_conv, detail=detail)


def _graph_flops(graph, only_conv=True, detail=False):
    assert isinstance(graph, GraphWrapper)
    flops = 0
    params2flops = {}
    for op in graph.ops():
        if op.type() in ['conv2d', 'depthwise_conv2d']:
            filter_shape = op.inputs("Filter")[0].shape()
            output_shape = op.outputs("Output")[0].shape()
            c_out, c_in, k_h, k_w = filter_shape
            _, _, h_out, w_out = output_shape
            # c_in is the channel number of filter. It is (input_channel // groups).
            kernel_ops = k_h * k_w * float(c_in)
            if len(op.inputs("Bias")) > 0:
                with_bias = 1
            else:
                with_bias = 0
            op_flops = h_out * w_out * c_out * (kernel_ops + with_bias)
            flops += op_flops
            params2flops[op.inputs("Filter")[0].name()] = op_flops
        elif op.type() == 'pool2d' and not only_conv:
            output_shape = op.outputs("Out")[0].shape()
            _, c_out, h_out, w_out = output_shape
            k_size = op.attr("ksize")
            flops += h_out * w_out * c_out * (k_size[0]**2)

        elif op.type() == 'mul':
            x_shape = list(op.inputs("X")[0].shape())
            y_shape = op.inputs("Y")[0].shape()
            if x_shape[0] == -1:
                x_shape[0] = 1

            op_flops = x_shape[0] * x_shape[1] * y_shape[1]
            flops += op_flops
            params2flops[op.inputs("Y")[0].name()] = op_flops

        elif op.type() in ['relu', 'sigmoid', 'batch_norm', 'relu6'
                           ] and not only_conv:
            input_shape = list(op.inputs("X")[0].shape())
            if input_shape[0] == -1:
                input_shape[0] = 1
            flops += np.product(input_shape)

    if detail:
        return flops, params2flops
    else:
        return flops


def dygraph_flops(model, inputs, dtypes=None, only_conv=False, detail=False):
    """
    Compute the FLOPs of nn.Layer.
    Args:
      model(nn.Layer): The target model.
      inputs(list): The dummy inputs used for 'model.forward'. It can be:
                      1. list<int>|tuple<int>: means 'model.forward' accepts
                         only one variable as argument and the shape of
                         variable is 'inputs'.
                      2. list<list<list>>: means 'model.forward' accepts multiple
                         variables as arguments and the shapes of variables is 'inputs'.
                      3. others: 'inputs' will be used as argument list by calling
                         'model.forward(*inputs)'.
      dtypes(str|list<str>): It only used when 'inputs' is shape or shapes that means
                      data type of each input. None means all the inputs is 'float32'.
                      Default: None.
      only_conv(bool): Just return number of mul-adds in convolution and FC layer if `only_conv` is true.
                         default: True.
      detail(bool): Whether to return detail of each convolution layer.
    """

    program = dygraph2program(model, inputs)
    graph = GraphWrapper(program)
    return _graph_flops(graph, only_conv=only_conv, detail=detail)
