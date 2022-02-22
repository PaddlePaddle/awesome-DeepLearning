import os
import pickle
import numpy as np
import logging
from .pruning_plan import PruningPlan
from paddleslim.common import get_logger

__all__ = ["Pruner"]

_logger = get_logger(__name__, level=logging.INFO)


class Pruner(object):
    """
    Pruner used to resize or mask dimensions of variables.
    Args:
        model(paddle.nn.Layer): The target model to be pruned.
        input_shape(list<int>): The input shape of model. It is used to trace the graph of the model.
        opt(paddle.optimizer.Optimizer): The model's optimizer. Default: None.
    """

    def __init__(self, model, inputs, opt=None):
        self.model = model
        self.inputs = inputs
        self._var_shapes = {}
        for var in model.parameters():
            self._var_shapes[var.name] = var.shape
        self.plan = None
        self.opt = opt

    def status(self, data=None, eval_func=None, status_file=None):
        raise NotImplemented("status is not implemented")

    def prune_var(self, var_name, axis, pruned_ratio, apply="impretive"):
        raise NotImplemented("prune_var is not implemented")

    def prune_vars(self, ratios, axis, apply="impretive"):
        """
        Pruning variables by given ratios.
        Args:
            ratios(dict<str, float>): The key is the name of variable to be pruned and the
                                      value is the pruned ratio.
            axis(int): The dimension to be pruned on.

        Returns:
            plan(PruningPlan): The pruning plan.
        """
        axis = axis[0] if isinstance(axis, list) else axis
        global_plan = PruningPlan(self.model.full_name)
        for var, ratio in ratios.items():
            if not global_plan.contains(var, axis):
                plan = self.prune_var(var, axis, ratio, apply=None)
                global_plan.extend(plan)
        if apply == "lazy":
            global_plan.apply(self.model, lazy=True)
        elif apply == "impretive":
            global_plan.apply(self.model, lazy=False, opt=self.opt)
        self.plan = global_plan
        return global_plan
