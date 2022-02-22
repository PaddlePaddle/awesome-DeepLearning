"""Define some functions to collect ralated parameters into groups."""
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import copy
import logging
from ..core import GraphWrapper, VarWrapper
from ..common import get_logger
from .prune_worker import PRUNE_WORKER, UnsupportOpError

__all__ = [
    'PruningDetails', 'PruningCollection', 'PruningCollections',
    'StaticPruningCollections'
]

_logger = get_logger(__name__, level=logging.INFO)


class PruningDetails(object):
    """
    The description of one pruning operation.
    Args:
        var(VarWrapper): The variable to be pruned.
        axis(int): The axis to be pruned on.
        transform(dict): Information used to convert pruned indices of master
                         tensor to indices of current tensor.
        op(OpWrapper): The operator with current tensor as input.
        is_parameter(bool): whether the tensor is parameter. Default: True.
    """

    def __init__(self, var, axis, transform, op, is_parameter=True):
        assert (isinstance(var, VarWrapper),
                "name should be VarWrapper, but get type = ".format(type(var)))
        assert (isinstance(axis, int))
        self.name = var.name()
        self.var = var
        self.axis = axis
        self.transform = transform
        self.op = op
        self.is_parameter = is_parameter

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.axis != other.axis:
            return False
        if self.transform != other.transform:
            return False
        return True


class PruningCollection(object):
    """
    A group of pruning operations.
    
      conv1-->conv2-->batch_norm

    For the network defined above, if weight of conv1 is pruned on 0-axis,
    weight of'conv2' should be pruned on 1-axis. The pruning operations on 0-axis of
    'conv1' and those on 1-axis of 'conv2' is a collection. And the {'name': conv1.weight_name, 'axis': 0}
    is the master of current collection.
     
    Args:
        master(dict): The master pruning operation.
    """

    def __init__(self, master=None):
        self._master = master
        self.master_name = master['name']
        self.master_axis = master['axis']
        self._nodes = {}

    def variables(self):
        """
        Get all tensors to be pruned in current collection.
        Returns:
          list<str>: Names of tensor to be pruned.
        """
        return list(self._nodes.keys())

    def add(self, node):
        """
        Add a pruning operation into current collention.
        Args:
            node(PruningDetails): Pruning operation to be added into current collection.
        """
        assert (isinstance(node, PruningDetails))
        # the first added pruning operation will be master.
        self._master = {
            "name": node.name,
            "axis": node.aixs
        } if self._master is None else self._master
        if node.name not in self._nodes:
            self._nodes[node.name] = []
        if node not in self._nodes[node.name]:
            self._nodes[node.name].append(node)

    @property
    def master(self):
        return self._master

    def all_pruning_details(self):
        """
        Get all pruning operations in current collection.
        Returns:
            list<PruningDetails>: Pruning operations.
        """
        ret = []
        for _items in self._nodes.values():
            ret.extend(_items)
        return ret


class PruningCollections(object):
    def __init__(self):
        self._collections = None

    def __iter__(self):
        return iter(self._collections)

    def _find_leaves(self, graph):
        ret = []
        for _var in graph.vars():
            if len(_var.outputs()) == 0:
                ret.append(_var.name())
        return ret

    def create_pruning_collections(self,
                                   params,
                                   graph,
                                   skip_stranger=True,
                                   skip_vars=None):
        """Collect convolution layers of graph into groups. The layers in the same group is relative on pruning operation.
        A group is a list of tuple with format (param_name, axis) in which `param_name` is the name of parameter and `axis` is the axis to be pruned on.
    
        .. code-block:: text
    
           conv1->conv2->conv3->conv4
    
        As shown above, the demo has 4 convolution layers. And the shape of convolution's parameter is `[out_channel, in_channel, filter_size, filter_size]`. If parameter of `conv1` was pruned on axis 0, then the parameter of `conv2` should be pruned on axis 1. So the `conv1` and `conv2` is a group that can be represented as:
    
        .. code-block:: python
    
           [("conv1", 0), ("conv2", 1)]
    
        If `params` is `["conv1", "conv2"]`, then the returned groups is:
    
        .. code-block:: python
    
           [[("conv1", 0), ("conv2", 1)],
            [("conv2", 0), ("conv3", 1)]]
    
        Args:
           params(list): A list of convolution layer's parameter names. It will collect all the groups that contains anyone of these parameters.
           graph(paddle.static.Program | GraphWrapper): The graph used to search the groups.
           skip_stranger(bool): Whether to skip current tensor when visit unregistered operators that not in OPS_UNCHANGE_SHAPE. False means visit all unregistered operators by default worker. Default: True.
           skip_vars(list<str>): Names of variables that will be skipped. None means skipping all leaves in given graph. '[]' means skipping nothing. Default: None.
    
        Returns:
           list<Group>: The groups.
    
        """
        if not isinstance(graph, GraphWrapper):
            graph = GraphWrapper(graph)

        if skip_vars is None:
            skip_vars = self._find_leaves(graph)
            _logger.warning(
                "Leaves {} will be skipped when parsing graph. You can set skipped variables by option 'skip_vars'.".
                format(skip_vars))

        visited = {}
        collections = []
        unsupported_warnings = set()
        for _param in params:
            pruned_params = []
            param = graph.var(_param)
            if param is None:
                _logger.warning(
                    f"Couldn't find relative variables of {_param} because {_param} is not in target program or model. Please make sure {_param} is in your program if you are using static API of PaddlePaddle. And make sure your model in correct mode and contains {_param} if you are using dynamic API of PaddlePaddle."
                )
                continue
            target_op = param.outputs()[0]
            if target_op.type() == 'conditional_block':
                for op in param.outputs():
                    if op.type() in PRUNE_WORKER._module_dict.keys():
                        cls = PRUNE_WORKER.get(op.type())
                        worker = cls(op,
                                     pruned_params=pruned_params,
                                     visited=visited,
                                     skip_stranger=skip_stranger)
            else:
                cls = PRUNE_WORKER.get(target_op.type())
                if cls is None:
                    _logger.warning("No worker for operator: {}".format(
                        target_op.type()))
                    continue
                worker = cls(target_op,
                             pruned_params=pruned_params,
                             visited=visited,
                             skip_stranger=skip_stranger)
                worker.skip_vars = skip_vars
            try:
                visited_backup = copy.deepcopy(worker.visited)
                worker.prune(param, pruned_axis=0, pruned_idx=[])
            except UnsupportOpError as e:
                visited.clear()
                visited.update(visited_backup)
                unsupported_warnings.add(e.args)
            else:
                if len(pruned_params) != 0:
                    collection = PruningCollection(master=({
                        "name": param.name(),
                        "axis": 0
                    }))
                    for _param, _axis, _transform, _op in pruned_params:
                        collection.add(
                            PruningDetails(_param, _axis, _transform, _op))
                    collections.append(collection)
        for warn in unsupported_warnings:
            _logger.warning(warn)
        self._collections = collections
        return self._collections


class StaticPruningCollections(PruningCollections):
    def __init__(self, params, graph, skip_stranger=True):
        super(StaticPruningCollections, self).__init__()
        self._collections = self.create_pruning_collections(
            params, graph, skip_stranger=skip_stranger)
