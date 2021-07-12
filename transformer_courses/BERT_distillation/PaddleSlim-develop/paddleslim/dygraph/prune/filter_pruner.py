import os
import logging
import numpy as np
import pickle
import copy
import paddle
from paddleslim.common import get_logger
from .var_group import *
from .pruning_plan import *
from .pruner import Pruner
from paddleslim.analysis import dygraph_flops as flops
from .var_group import DygraphPruningCollections

__all__ = ['Status', 'FilterPruner']

_logger = get_logger(__name__, logging.INFO)

CONV_OP_TYPE = paddle.nn.Conv2D
FILTER_DIM = 0
CONV_WEIGHT_NAME = "weight"
SKIP_LAYERS = (paddle.nn.Conv2DTranspose, paddle.nn.layer.conv.Conv2DTranspose)


class Status():
    def __init__(self, src=None):
        self.sensitivies = {}
        self.accumulates = {}
        self.is_ckp = True
        if src is not None:
            self.load(src)

    def save(self, dst):
        with open(dst, 'wb') as f:
            pickle.dump(self, f)
            _logger.info("Save status into {}".format(dst))

    def load(self, src):
        with open(src, 'rb') as f:
            data = pickle.load(f)
            self.sensitivies = data.sensitivies
            self.accumulates = data.accumulates
            self.is_ckp = data.is_ckp
            _logger.info("Load status from {}".format(src))


class FilterPruner(Pruner):
    """
    Pruner used to prune filter structure in convolution layer.

    Args:
        model(paddle.nn.Layer): The target model to be pruned.
        inputs(list<Object>): The inputs of model. It will be use in calling 'model.forward(inputs)'.
        sen_file(str, optional): The absolute path of file that stores computed sensitivities. If it is
                              set rightly, 'FilterPruner::sensitive' function can not be called anymore
                              in next step. Default: None.
    
    """

    def __init__(self, model, inputs, sen_file=None, opt=None):
        super(FilterPruner, self).__init__(model, inputs, opt=opt)
        self._status = Status(sen_file)
        # sensitive and collections are just used in filter pruning
        self.collections = DygraphPruningCollections(model, inputs)
        # skip vars in:
        # 1. depthwise conv2d layer
        self.skip_vars = []
        for sub_layer in model.sublayers():
            #if isinstance(sub_layer, SKIP_LAYERS) or (isinstance(
            #        sub_layer, paddle.nn.layer.conv.Conv2D) and
            #                                          sub_layer._groups > 1):
            if isinstance(sub_layer, SKIP_LAYERS):
                for param in sub_layer.parameters():
                    self.skip_vars.append(param.name)

    def sensitive(self,
                  eval_func=None,
                  sen_file=None,
                  target_vars=None,
                  skip_vars=[]):
        """
        Compute or get sensitivities of model in current pruner. It will return a cached sensitivities when all the arguments are "None".

        This function return a dict storing sensitivities as below:
    
        .. code-block:: python
    
               {"weight_0":
                   {0.1: 0.22,
                    0.2: 0.33
                   },
                 "weight_1":
                   {0.1: 0.21,
                    0.2: 0.4
                   }
               }
    
        ``weight_0`` is parameter name of convolution. ``sensitivities['weight_0']`` is a dict in which key is pruned ratio and value is the percent of losses.

        Args:
          eval_func(function, optional): The function to evaluate the model in current pruner. This function should have an empy arguments list and return a score with type "float32". Default: None.
          sen_file(str, optional): The absolute path of file to save sensitivities into local filesystem. Default: None.
          target_vars(list, optional): The names of tensors whose sensitivity will be computed. "None" means all weights in convolution layer will be computed. Default: None.
          skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. Default: [].
    
        Returns:
           dict: A dict storing sensitivities.       

        """
        if eval_func is None and sen_file is None:
            return self._status.sensitivies
        if sen_file is not None and os.path.isfile(sen_file):
            self._status.load(sen_file)

        if not self._status.is_ckp:
            return self._status

        skip_vars.extend(self.skip_vars)
        self._cal_sensitive(
            self.model,
            eval_func,
            status_file=sen_file,
            target_vars=target_vars,
            skip_vars=skip_vars)

        self._status.is_ckp = False
        return self._status.sensitivies

    def _get_ratios_by_loss(self, sensitivities, loss, skip_vars=[]):
        """
        Get the max ratio of each parameter. The loss of accuracy must be less than given `loss`
        when the single parameter was pruned by the max ratio. 
        
        Args:
          
          sensitivities(dict): The sensitivities used to generate a group of pruning ratios. The key of dict
                               is name of parameters to be pruned. The value of dict is a list of tuple with
                               format `(pruned_ratio, accuracy_loss)`.
          loss(float): The threshold of accuracy loss.
          skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. "None" means skip nothing. Default: None.
    
        Returns:
    
          dict: A group of ratios. The key of dict is name of parameters while the value is the ratio to be pruned.
        """
        ratios = {}
        for param, losses in sensitivities.items():
            if param in skip_vars:
                continue
            losses = losses.items()
            losses = list(losses)
            losses.sort()
            for i in range(len(losses))[::-1]:
                if losses[i][1] <= loss:
                    if i == (len(losses) - 1):
                        ratios[param] = float(losses[i][0])
                    else:
                        r0, l0 = losses[i]
                        r1, l1 = losses[i + 1]
                        r0 = float(r0)
                        r1 = float(r1)
                        d0 = loss - l0
                        d1 = l1 - loss

                        ratio = r0 + (loss - l0) * (r1 - r0) / (l1 - l0)
                        ratios[param] = ratio
                        if ratio > 1:
                            _logger.info(losses, ratio, (r1 - r0) / (l1 - l0),
                                         i)

                    break
        return ratios

    def _round_to(self, ratios, dims=0, factor=8):
        ret = {}
        for name in ratios:
            ratio = ratios[name]
            dim = self._var_shapes[name][dims]
            remained = round((1 - ratio) * dim / factor) * factor
            if remained == 0:
                remained = factor
            ratio = float(dim - remained) / dim
            ratio = ratio if ratio > 0 else 0.
            ret[name] = ratio
        return ret

    def get_ratios_by_sensitivity(self,
                                  pruned_flops,
                                  align=None,
                                  dims=0,
                                  skip_vars=[]):
        """
         Get a group of ratios by sensitivities.
         Args:
             pruned_flops(float): The excepted rate of FLOPs to be pruned. It should be in range (0, 1).
             align(int, optional): Round the size of each pruned dimension to multiple of 'align' if 'align' is not None. Default: None.
             dims(int, optional): The dims to be pruned on. 0 means pruning channels of output for convolution. Default: 0.
             skip_vars(list, optional): The names of tensors whose sensitivity won't be computed. "None" means skip nothing. Default: None.

        Returns:
            tuple: A tuple with format ``(ratios, pruned_flops)`` . "ratios" is a dict whose key is name of tensor and value is ratio to be pruned. "pruned_flops" is the ratio of total pruned FLOPs in the model.
        """
        base_flops = flops(self.model, self.inputs)

        _logger.info("Base FLOPs: {}".format(base_flops))
        low = 0.
        up = 1.0
        history = set()
        while low < up:
            loss = (low + up) / 2
            ratios = self._get_ratios_by_loss(
                self._status.sensitivies, loss, skip_vars=skip_vars)
            _logger.debug("pruning ratios: {}".format(ratios))
            if align is not None:
                ratios = self._round_to(ratios, dims=dims, factor=align)
            plan = self.prune_vars(ratios, axis=dims)
            c_flops = flops(self.model, self.inputs)
            c_pruned_flops = (base_flops - c_flops) / base_flops
            plan.restore(self.model, opt=self.opt)
            _logger.debug("Seaching ratios, pruned FLOPs: {}".format(
                c_pruned_flops))
            key = str(round(c_pruned_flops, 4))
            if key in history:
                return ratios, c_pruned_flops
            history.add(key)
            if c_pruned_flops < pruned_flops:
                low = loss
            elif c_pruned_flops > pruned_flops:
                up = loss
            else:
                return ratios, c_pruned_flops
        return ratios, c_pruned_flops

    def _cal_sensitive(self,
                       model,
                       eval_func,
                       status_file=None,
                       target_vars=None,
                       skip_vars=None):
        sensitivities = self._status.sensitivies
        baseline = None
        ratios = np.arange(0.1, 1, step=0.1)
        for _collection in self.collections:
            var_name = _collection.master_name
            dims = _collection.master_axis
            if target_vars is not None and var_name not in target_vars:
                continue
            if skip_vars is not None and var_name in skip_vars:
                continue

            if var_name not in sensitivities:
                sensitivities[var_name] = {}
            for ratio in ratios:
                ratio = round(ratio, 2)
                if ratio in sensitivities[var_name]:
                    _logger.debug("{}, {} has computed.".format(var_name,
                                                                ratio))
                    continue
                if baseline is None:
                    baseline = eval_func()
                plan = self.prune_var(var_name, dims, ratio)
                pruned_metric = eval_func()
                loss = (baseline - pruned_metric) / baseline
                _logger.info("pruned param: {}; {}; loss={}".format(
                    var_name, ratio, loss))
                sensitivities[var_name][ratio] = loss
                self._status.save(status_file)
                plan.restore(model, opt=self.opt)

        return sensitivities

    def sensitive_prune(self, pruned_flops, skip_vars=[], align=None):

        # skip depthwise convolutions
        for layer in self.model.sublayers():
            if isinstance(layer,
                          paddle.nn.layer.conv.Conv2D) and layer._groups > 1:
                for param in layer.parameters(include_sublayers=False):
                    skip_vars.append(param.name)
        _logger.debug("skip vars: {}".format(skip_vars))
        self.restore()
        ratios, pruned_flops = self.get_ratios_by_sensitivity(
            pruned_flops, align=align, dims=FILTER_DIM, skip_vars=skip_vars)
        self.plan = self.prune_vars(ratios, FILTER_DIM)
        self.plan._pruned_flops = pruned_flops
        return self.plan

    def restore(self):
        if self.plan is not None:
            self.plan.restore(self.model, opt=self.opt)

    def cal_mask(self, pruned_ratio, collection):
        raise NotImplemented("cal_mask is not implemented")

    def prune_var(self, var_name, pruned_axis, pruned_ratio, apply="impretive"):
        """
        Pruning a variable.
        Parameters:
            var_name(str): The name of variable.
            pruned_axis(int): The axis to be pruned. For convolution with format [out_c, in_c, k, k],
                             'axis=0' means pruning filters.
            pruned_ratio(float): The ratio of pruned values in one variable.
            apply(str): How to apply pruning plan to graph. It can be 'impretive', 'lazy' or None. None
                        means just returning an instance of 'PruningPlan' but not applying it to graph.

        Returns:
            plan: An instance of PruningPlan that can be applied on model by calling 'plan.apply(model)'.

        """
        pruned_axis = pruned_axis[0] if isinstance(pruned_axis,
                                                   list) else pruned_axis
        assert (isinstance(pruned_axis, int))
        if var_name in self.skip_vars:
            _logger.warn(
                f"{var_name} is skiped beacause it is not supported for pruning directly."
            )
            return
        collection = self.collections.find_collection_by_master(var_name,
                                                                pruned_axis)
        plan = PruningPlan(self.model.full_name)
        if collection is None:
            _logger.debug(
                f"Can not find collection with master ['name': {var_name}, 'axis': {pruned_axis}]"
            )
            return plan
        _logger.info(
            f"Pruning variable [{var_name}] and its relatives {list(collection.variables())}"
        )

        mask = self.cal_mask(pruned_ratio, collection)
        for _detail in collection.all_pruning_details():
            # Varibales can be pruned on multiple axies. 
            src_mask = copy.deepcopy(mask)
            var_shape = _detail.var.shape()
            for tran in _detail.transform:

                src_mask = self._transform_mask(src_mask, tran)
            current_mask = src_mask
            groups = _detail.op.attr('groups')
            if groups is None or groups == 1:
                assert len(current_mask) == var_shape[
                    _detail.
                    axis], f"The length of current_mask must be equal to the size of dimension to be pruned on. But get: len(current_mask): {len(current_mask)}; var_shape: {var_shape}; axis: {_detail.axis}; var name: {_detail.name}; len(mask): {len(mask)}"
            plan.add(_detail.name,
                     PruningMask(_detail.axis, current_mask, pruned_ratio,
                                 _detail.op))
        if apply == "lazy":
            plan.apply(self.model, lazy=True)
        elif apply == "impretive":
            plan.apply(self.model, lazy=False, opt=self.opt)
        return plan

    def _transform_mask(self, mask, transform):
        if "src_start" in transform:
            src_start = transform['src_start']
            src_end = transform['src_end']
            target_start = transform['target_start']
            target_end = transform['target_end']
            target_len = transform['target_len']
            mask = mask[src_start:src_end]
            dst_mask = np.ones([target_len])
            expand = int((target_end - target_start) / len(mask))
            dst_mask[target_start:target_end] = list(mask) * expand
        elif "stride" in transform:
            stride = transform['stride']
            mask = mask.repeat(stride) if stride > 1 else mask
            return mask
        else:
            return mask
        return dst_mask
