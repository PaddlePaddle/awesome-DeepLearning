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

import logging
import sys
import copy
import numpy as np
from functools import reduce
from ..core import VarWrapper, OpWrapper, GraphWrapper
from .collections import StaticPruningCollections
from .criterion import CRITERION
from .idx_selector import IDX_SELECTOR
from ..common import get_logger

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner():
    """The pruner used to prune channels of convolution.

    Args:
        criterion(str|function): the criterion used to sort channels for pruning.
        idx_selector(str|function): 

    """

    def __init__(self, criterion="l1_norm",
                 idx_selector="default_idx_selector"):
        if isinstance(criterion, str):
            self.criterion = CRITERION.get(criterion)
        else:
            self.criterion = criterion
        if isinstance(idx_selector, str):
            self.idx_selector = IDX_SELECTOR.get(idx_selector)
        else:
            self.idx_selector = idx_selector

        self.pruned_weights = False

    def prune(self,
              program,
              scope,
              params,
              ratios,
              place=None,
              lazy=False,
              only_graph=False,
              param_backup=False,
              param_shape_backup=False):
        """Pruning the given parameters.

        Args:

            program(paddle.static.Program): The program to be pruned.
            scope(paddle.static.Scope): The scope storing paramaters to be pruned.
            params(list<str>): A list of parameter names to be pruned.
            ratios(list<float>): A list of ratios to be used to pruning parameters.
            place(paddle.CUDAPlace||paddle.CPUPlace): The device place of filter parameters. Defalut: None.
            lazy(bool): True means setting the pruned elements to zero.
                        False means cutting down the pruned elements. Default: False.
            only_graph(bool): True means only modifying the graph.
                              False means modifying graph and variables in scope. Default: False.
            param_backup(bool): Whether to return a dict to backup the values of parameters. Default: False.
            param_shape_backup(bool): Whether to return a dict to backup the shapes of parameters. Default: False.

        Returns:
            tuple: ``(pruned_program, param_backup, param_shape_backup)``. ``pruned_program`` is the pruned program. ``param_backup`` is a dict to backup the values of parameters. ``param_shape_backup`` is a dict to backup the shapes of parameters.
        """
        self.pruned_list = []
        graph = GraphWrapper(program.clone())
        param_backup = {} if param_backup else None
        param_shape_backup = {} if param_shape_backup else None

        pruned_params = []
        collections = StaticPruningCollections(params, graph)
        ratios = dict(zip(params, ratios))
        values = {}
        for _collection in collections:
            for _var_name in _collection.variables():
                var = scope.find_var(_var_name)
                if var is not None:
                    value = np.array(var.get_tensor())
                    values[_var_name] = value

        for _collection in collections:
            scores = self.criterion(_collection, values, graph)
            idx = self.idx_selector(_collection, scores,
                                    ratios)  # name, axis, idx, transform
            idx = self._transform(idx)
            pruned_params.extend(idx)

        merge_pruned_params = {}
        for param, pruned_axis, pruned_idx in pruned_params:
            if param not in merge_pruned_params:
                merge_pruned_params[param] = {}
            if pruned_axis not in merge_pruned_params[param]:
                merge_pruned_params[param][pruned_axis] = []
            merge_pruned_params[param][pruned_axis].append(pruned_idx)
        for param_name in merge_pruned_params:
            for pruned_axis in merge_pruned_params[param_name]:
                pruned_idx = np.concatenate(merge_pruned_params[param_name][
                    pruned_axis])
                param = graph.var(param_name)
                _groups = 1
                if not lazy:
                    # update groups of conv2d
                    if pruned_axis == 1:
                        for op in param.outputs():
                            if op.type() in ["conv2d", "depthwise_conv2d"
                                             ] and op.attr("groups") > 1:
                                _groups = op.attr("groups")
                                _filter_num = param.shape()[1]
                                new_groups = int(
                                    (_groups * _filter_num - len(pruned_idx)) /
                                    _filter_num)
                                _logger.info(
                                    f"change groups of {op.type()}({param.name()}) from {op.attr('groups')} to {new_groups};"
                                )
                                op.set_attr("groups", new_groups)
                    if _groups == 1:
                        origin_shape = copy.deepcopy(param.shape())
                        if param_shape_backup is not None:
                            param_shape_backup[param.name()] = origin_shape
                        new_shape = list(param.shape())
                        new_shape[pruned_axis] -= len(pruned_idx)
                        param.set_shape(new_shape)

                if not only_graph and (_groups == 1 or pruned_axis != 1):
                    _var = scope.find_var(param.name())
                    if _var is None:
                        continue
                    param_t = _var.get_tensor()
                    if param_backup is not None and (
                            param.name() not in param_backup):
                        param_backup[param.name()] = copy.deepcopy(
                            np.array(param_t))
                    try:
                        pruned_param = self._prune_tensor(
                            np.array(param_t),
                            pruned_idx,
                            pruned_axis=pruned_axis,
                            lazy=lazy)
                        param_t.set(pruned_param, place)
                    except IndexError as e:
                        _logger.error(
                            "Pruning {} with shape {} on axis {}, but get [{}]; ".
                            format(param.name(),
                                   param_t.shape(), pruned_axis, e))

        graph.infer_shape()
        self.pruned_weights = (not only_graph)
        return graph.program, param_backup, param_shape_backup

    def _transform(self, items):
        ret = []
        for name, axis, pruned_idx, transforms in items:
            src = pruned_idx
            for trans in transforms:
                if 'src_start' not in trans:
                    continue
                src_start = trans['src_start']
                src_end = trans['src_end']
                src_len = src_end - src_start
                target_start = trans['target_start']
                target_end = trans['target_end']
                starts = np.array(range(target_start, target_end, src_len))
                target = []
                for idx in src:
                    if idx >= src_start and idx < src_end:
                        idx -= src_start
                        target.extend(list(idx + starts))
                src = target
            ret.append((name, axis, src))
        return ret

    def _prune_tensor(self, tensor, pruned_idx, pruned_axis, lazy=False):
        """
        Pruning a array by indices on given axis.

        Args:
            tensor(numpy.array): The target array to be pruned.
            pruned_idx(list<int>): The indices to be pruned.
            pruned_axis(int): The axis of given array to be pruned on. 
            lazy(bool): True means setting the pruned elements to zero.
                        False means remove the pruned elements from memory.
                        default: False.

        Returns:
            numpy.array: The pruned array.
        """
        mask = np.zeros(tensor.shape[pruned_axis], dtype=bool)
        mask[pruned_idx] = True

        def func(data):
            return data[~mask]

        def lazy_func(data):
            data[mask] = 0
            return data

        if lazy:
            return np.apply_along_axis(lazy_func, pruned_axis, tensor)
        else:
            return np.apply_along_axis(func, pruned_axis, tensor)
