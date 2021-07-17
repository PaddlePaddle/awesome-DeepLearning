#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import paddle
from paddle.fluid import core
from .layers_base import BaseBlock

__all__ = ['get_prune_params_config', 'prune_params', 'check_search_space']

WEIGHT_OP = [
    'conv2d', 'linear', 'embedding', 'conv2d_transpose', 'depthwise_conv2d'
]
CONV_TYPES = [
    'conv2d', 'conv3d', 'conv1d', 'superconv2d', 'supergroupconv2d',
    'superdepthwiseconv2d'
]


def get_prune_params_config(graph, origin_model_config):
    """ Convert config of search space to parameters' prune config.
    """
    param_config = {}
    precedor = None
    for op in graph.ops():
        ### TODO(ceci3):
        ### 1. fix config when this op is concat by graph.pre_ops(op)
        ### 2. add kernel_size in config
        for inp in op.all_inputs():
            n_ops = graph.next_ops(op)
            if inp._var.name in origin_model_config.keys():
                if 'expand_ratio' in origin_model_config[
                        inp._var.name] or 'channel' in origin_model_config[
                            inp._var.name]:
                    key = 'channel' if 'channel' in origin_model_config[
                        inp._var.name] else 'expand_ratio'
                    tmp = origin_model_config[inp._var.name][key]
                    if len(inp._var.shape) > 1:
                        if inp._var.name in param_config.keys():
                            param_config[inp._var.name].append(tmp)
                        ### first op
                        else:
                            param_config[inp._var.name] = [precedor, tmp]
                    else:
                        param_config[inp._var.name] = [tmp]
                    precedor = tmp
                else:
                    precedor = None
            for n_op in n_ops:
                for next_inp in n_op.all_inputs():
                    if next_inp._var.persistable == True:
                        if next_inp._var.name in origin_model_config.keys():
                            if 'expand_ratio' in origin_model_config[
                                    next_inp._var.
                                    name] or 'channel' in origin_model_config[
                                        next_inp._var.name]:
                                key = 'channel' if 'channel' in origin_model_config[
                                    next_inp._var.name] else 'expand_ratio'
                                tmp = origin_model_config[next_inp._var.name][
                                    key]
                                pre = tmp if precedor is None else precedor
                                if len(next_inp._var.shape) > 1:
                                    param_config[next_inp._var.name] = [pre]
                                else:
                                    param_config[next_inp._var.name] = [tmp]
                            else:
                                if len(next_inp._var.
                                       shape) > 1 and precedor != None:
                                    param_config[
                                        next_inp._var.name] = [precedor, None]
                        else:
                            param_config[next_inp._var.name] = [precedor]

    return param_config


def get_actual_shape(transform, channel):
    if transform == None:
        channel = int(channel)
    else:
        if isinstance(transform, float):
            channel = int(channel * transform)
        else:
            channel = int(transform)
    return channel


def prune_params(model, param_config, super_model_sd=None):
    """ Prune parameters according to the config.
        Parameters:
            model(paddle.nn.Layer): instance of model.
            param_config(dict): prune config of each weight.
            super_model_sd(dict, optional): parameters come from supernet. If super_model_sd is not None, transfer parameters from this dict to model; otherwise, prune model from itself.
    """
    for l_name, sublayer in model.named_sublayers():
        if isinstance(sublayer, BaseBlock):
            continue
        for p_name, param in sublayer.named_parameters(include_sublayers=False):
            t_value = param.value().get_tensor()
            value = np.array(t_value).astype("float32")

            if super_model_sd != None:
                name = l_name + '.' + p_name
                super_t_value = super_model_sd[name].value().get_tensor()
                super_value = np.array(super_t_value).astype("float32")
                super_model_sd.pop(name)

            if param.name in param_config.keys():
                if len(param_config[param.name]) > 1:
                    in_exp = param_config[param.name][0]
                    out_exp = param_config[param.name][1]
                    if sublayer.__class__.__name__.lower() in CONV_TYPES:
                        in_chn = get_actual_shape(in_exp, value.shape[1])
                        out_chn = get_actual_shape(out_exp, value.shape[0])
                        prune_value = super_value[:out_chn, :in_chn, ...] \
                                         if super_model_sd != None else value[:out_chn, :in_chn, ...]
                    else:
                        in_chn = get_actual_shape(in_exp, value.shape[0])
                        out_chn = get_actual_shape(out_exp, value.shape[1])
                        prune_value = super_value[:in_chn, :out_chn, ...] \
                                         if super_model_sd != None else value[:in_chn, :out_chn, ...]
                else:
                    out_chn = get_actual_shape(param_config[param.name][0],
                                               value.shape[0])
                    prune_value = super_value[:out_chn, ...] \
                                     if super_model_sd != None else value[:out_chn, ...]

            else:
                prune_value = super_value if super_model_sd != None else value

            p = t_value._place()
            if p.is_cpu_place():
                place = core.CPUPlace()
            elif p.is_cuda_pinned_place():
                place = core.CUDAPinnedPlace()
            else:
                place = core.CUDAPlace(p.gpu_device_id())
            t_value.set(prune_value, place)
            if param.trainable:
                param.clear_gradient()

    ### initialize param which not in sublayers, such as create persistable inputs by create_parameters
    if super_model_sd != None and len(super_model_sd) != 0:
        for k, v in super_model_sd.items():
            setattr(model, k, v)


def _is_depthwise(op):
    """Check if this op is depthwise conv. Only Cin == Cout == groups be consider as depthwise conv.
       The shape of input and the shape of output in depthwise conv must be same in superlayer,
       so depthwise op cannot be consider as weight op
    """
    #if op.type() == 'depthwise_conv2d': ### depthwise_conv2d in paddle is Cout % Cin =0
    #    return True
    if 'conv' in op.type():
        for inp in op.all_inputs():
            if inp._var.persistable and (
                    op.attr('groups') == inp._var.shape[0] and
                    op.attr('groups') * inp._var.shape[1] == inp._var.shape[0]):
                return True
    return False


def _find_weight_ops(op, graph, weights):
    """ Find the vars come from operators with weight.
    """
    pre_ops = graph.pre_ops(op)
    for pre_op in pre_ops:
        ### if depthwise conv is one of elementwise's input, 
        ### add it into this same search space
        if _is_depthwise(pre_op):
            for inp in pre_op.all_inputs():
                if inp._var.persistable:
                    weights.append(inp._var.name)

        if pre_op.type() in WEIGHT_OP and not _is_depthwise(pre_op):
            for inp in pre_op.all_inputs():
                if inp._var.persistable:
                    weights.append(inp._var.name)
            return weights
        return _find_weight_ops(pre_op, graph, weights)
    return weights


def _find_pre_elementwise_add(op, graph):
    """ Find precedors of the elementwise_add operator in the model.
    """
    same_prune_before_elementwise_add = []
    pre_ops = graph.pre_ops(op)
    for pre_op in pre_ops:
        if pre_op.type() in WEIGHT_OP:
            return
        same_prune_before_elementwise_add = _find_weight_ops(
            pre_op, graph, same_prune_before_elementwise_add)
    return same_prune_before_elementwise_add


def check_search_space(graph):
    """ Find the shortcut in the model and set same config for this situation.
    """
    same_search_space = []
    depthwise_conv = []
    for op in graph.ops():
        if op.type() == 'elementwise_add' or op.type() == 'elementwise_mul':
            inp1, inp2 = op.all_inputs()[0], op.all_inputs()[1]
            if (not inp1._var.persistable) and (not inp2._var.persistable):
                pre_ele_op = _find_pre_elementwise_add(op, graph)
                if pre_ele_op != None:
                    same_search_space.append(pre_ele_op)

        if _is_depthwise(op):
            for inp in op.all_inputs():
                if inp._var.persistable:
                    depthwise_conv.append(inp._var.name)

    if len(same_search_space) == 0:
        return None, []

    same_search_space = sorted([sorted(x) for x in same_search_space])
    final_search_space = []

    if len(same_search_space) >= 1:
        final_search_space = [same_search_space[0]]
        if len(same_search_space) > 1:
            for l in same_search_space[1:]:
                listset = set(l)
                merged = False
                for idx in range(len(final_search_space)):
                    rset = set(final_search_space[idx])
                    if len(listset & rset) != 0:
                        final_search_space[idx] = list(listset | rset)
                        merged = True
                        break
                if not merged:
                    final_search_space.append(l)
    final_search_space = sorted([sorted(x) for x in final_search_space])
    depthwise_conv = sorted(depthwise_conv)

    return (final_search_space, depthwise_conv)


def broadcast_search_space(same_search_space, param2key, origin_config):
    """
    Inplace broadcast the origin_config according to the same search space. Such as: same_search_space = [['conv1_weight', 'conv3_weight']], param2key = {'conv1_weight': 'conv1.conv', 'conv3_weight': 'conv3.weight'}, origin_config= {'conv1.weight': {'channel': 10}, 'conv2.weight': {'channel': 20}}, the result after this function is origin_config={'conv1.weight': {'channel': 10}, 'conv2.weight': {'channel': 20}, 'conv3.weight': {'channel': 10}}

    Args:
        same_search_space(list<list>): broadcast according this list, each list in same_search_space means the channel must be consistent.
        param2key(dict): the name of layers corresponds to the name of parameter.
        origin_config(dict): the search space which can be searched.
    """
    for per_ss in same_search_space:
        for ss in per_ss[1:]:
            key = param2key[ss]
            pre_key = param2key[per_ss[0]]
            if key in origin_config:
                if 'expand_ratio' in origin_config[pre_key]:
                    origin_config[key].update({
                        'expand_ratio': origin_config[pre_key]['expand_ratio']
                    })
                elif 'channel' in origin_config[pre_key]:
                    origin_config[key].update({
                        'channel': origin_config[pre_key]['channel']
                    })
            else:
                if 'expand_ratio' in origin_config[pre_key]:
                    origin_config[key] = {
                        'expand_ratio': origin_config[pre_key]['expand_ratio']
                    }
                elif 'channel' in origin_config[pre_key]:
                    origin_config[key] = {
                        'channel': origin_config[pre_key]['channel']
                    }
