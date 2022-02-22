# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import copy
import pickle
import numpy as np
from collections import OrderedDict
from collections import Iterable
from paddle.fluid.framework import Program, program_guard, Parameter, Variable

__all__ = ['GraphWrapper', 'VarWrapper', 'OpWrapper']

OPTIMIZER_OPS = [
    'momentum',
    'lars_momentum',
    'adagrad',
    'adam',
    'adamax',
    'dpsgd',
    'decayed_adagrad',
    'adadelta',
    'rmsprop',
]


class VarWrapper(object):
    def __init__(self, var, graph):
        assert isinstance(var, Variable)
        assert isinstance(graph, GraphWrapper)
        self._var = var
        self._graph = graph

    def __eq__(self, v):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return (v is not None) and self._var.name == v._var.name

    def name(self):
        """
        Get the name of the variable.
        """
        return self._var.name

    def __repr__(self):
        return self._var.name

    def shape(self):
        """
        Get the shape of the varibale.
        """
        return self._var.shape

    def set_shape(self, shape):
        """
        Set the shape of the variable.
        """
        self._var.desc.set_shape(shape)

    def inputs(self):
        """
        Get all the operators that use this variable as output.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for op in self._graph.ops():
            if self in op.all_outputs():
                ops.append(op)
        return ops

    def outputs(self):
        """
        Get all the operators that use this variable as input.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for op in self._graph.ops():
            if self in op.all_inputs():
                ops.append(op)
        return ops

    def is_parameter(self):
        return isinstance(self._var, Parameter)


class OpWrapper(object):
    def __init__(self, op, graph):
        assert isinstance(graph, GraphWrapper)
        self._op = op
        self._graph = graph

    def __eq__(self, op):
        """
        Overwrite this function for ...in... syntax in python.
        """
        return self.idx() == op.idx()

    def all_inputs(self):
        """
        Get all the input variables of this operator.
        """
        return [
            self._graph.var(var_name) for var_name in self._op.input_arg_names
        ]

    def all_outputs(self):
        """
        Get all the output variables of this operator.
        """
        return [
            self._graph.var(var_name) for var_name in self._op.output_arg_names
        ]

    def idx(self):
        """
        Get the id of this operator.
        """
        return self._op.idx

    def type(self):
        """
        Get the type of this operator.
        """
        return self._op.type

    def __repr__(self):
        return "op[id: {}, type: {}; inputs: {}]".format(self.idx(),
                                                         self.type(),
                                                         self.all_inputs())

    def is_bwd_op(self):
        """
        Whether this operator is backward op.
        """
        return self.type().endswith('_grad')

    def is_opt_op(self):
        """
        Whether this operator is optimizer op.
        """
        return self.type() in OPTIMIZER_OPS

    def inputs(self, name):
        """
        Get all the varibales by the input name.
        """
        if name in self._op.input_names:
            return [
                self._graph.var(var_name) for var_name in self._op.input(name)
            ]
        return []

    def outputs(self, name):
        """
        Get all the varibales by the output name.
        """
        if name in self._op.output_names:
            return [
                self._graph.var(var_name) for var_name in self._op.output(name)
            ]
        return []

    def set_attr(self, key, value):
        """
        Set the value of attribute by attribute's name.

        Args:
            key(str): the attribute name.
            value(bool|int|str|float|list): the value of the attribute.
        """
        self._op._set_attr(key, value)

    def attr(self, name):
        """
        Get the attribute by name.

        Args:
            name(str): the attribute name.

        Returns:
            bool|int|str|float|list: The attribute value. The return value
            can be any valid attribute type.
        """
        if self._op.has_attr(name):
            return self._op.attr(name)
        else:
            return None


class GraphWrapper(object):
    """
    It is a wrapper of paddle.fluid.framework.IrGraph with some special functions
    for paddle slim framework.

    Args:
        program(framework.Program): A program with 
        in_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
        out_nodes(dict): A dict to indicate the input nodes of the graph.
                        The key is user-defined and human-readable name.
                        The value is the name of Variable.
    """

    def __init__(self, program=None, in_nodes=[], out_nodes=[]):
        """
        """
        super(GraphWrapper, self).__init__()
        self.program = Program() if program is None else program
        self.persistables = {}
        self.teacher_persistables = {}
        for var in self.program.list_vars():
            if var.persistable:
                self.persistables[var.name] = var
        self.compiled_graph = None
        in_nodes = [] if in_nodes is None else in_nodes
        out_nodes = [] if out_nodes is None else out_nodes
        self.in_nodes = OrderedDict(in_nodes)
        self.out_nodes = OrderedDict(out_nodes)
        self._attrs = OrderedDict()

    def all_parameters(self):
        """
        Get all the parameters in this graph.

        Returns:
            list<VarWrapper>: A list of VarWrapper instances.
        """
        params = []
        for block in self.program.blocks:
            for param in block.all_parameters():
                params.append(VarWrapper(param, self))
        return params

    def is_parameter(self, var):
        """
        Whether the given variable is parameter.

        Args:
            var(VarWrapper): The given varibale.
        """
        return isinstance(var._var, Parameter)

    def is_persistable(self, var):
        """
        Whether the given variable is persistable.

        Args:
            var(VarWrapper): The given varibale.
        """
        return var._var.persistable

    def ops(self):
        """
        Return all operator nodes included in the graph as a set.
        """
        ops = []
        for block in self.program.blocks:
            for op in block.ops:
                ops.append(OpWrapper(op, self))
        return ops

    def vars(self):
        """
        Get all the variables.
        """
        return [VarWrapper(var, self) for var in self.program.list_vars()]

    def var(self, name):
        """
        Get the variable by variable name.
        """
        for block in self.program.blocks:
            if block.has_var(name):
                return VarWrapper(block.var(name), self)
        return None

    def clone(self, for_test=False):
        """
        Clone a new graph from current graph.

        Returns:
            (GraphWrapper): The wrapper of a new graph.
        """
        return GraphWrapper(
            self.program.clone(for_test),
            copy.deepcopy(self.in_nodes), copy.deepcopy(self.out_nodes))

    def program(self):
        """
        Get the program in current wrapper.
        """
        return self.program

    def pre_ops(self, op):
        """
        Get all the previous operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for in_var in op.all_inputs():
                if in_var in p.all_outputs():
                    ops.append(p)
        return ops

    def next_ops(self, op):
        """
        Get all the next operators of target operator.

        Args:
            op(OpWrapper): Target operator.

        Returns:
            list<OpWrapper>: A list of operators.
        """
        ops = []
        for p in self.ops():
            for out_var in op.all_outputs():
                if out_var in p.all_inputs():
                    ops.append(p)
        return ops

    def get_param_by_op(self, op):
        """
        Get the parameters used by target operator.
        """
        assert isinstance(op, OpWrapper)
        params = []
        for var in op.all_inputs():
            if isinstance(var._var, Parameter):
                params.append(var)
        assert len(params) > 0
        return params

    def numel_params(self):
        """
        Get the number of elements in all parameters.
        """
        ret = 0
        for param in self.all_parameters():
            ret += np.product(param.shape())
        return ret

    def infer_shape(self):
        """
        Update the groups of convolution layer according to current filters.
        It is used after loading pruned parameters from file.
        """
        for op in self.ops():
            if op.type() != 'conditional_block':
                op._op.desc.infer_shape(op._op.block.desc)
