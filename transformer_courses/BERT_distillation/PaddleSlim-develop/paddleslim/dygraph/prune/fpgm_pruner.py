import logging
import numpy as np
import paddle
from paddleslim.common import get_logger
from .var_group import *
from .pruning_plan import *
from .filter_pruner import FilterPruner

__all__ = ['FPGMFilterPruner']

_logger = get_logger(__name__, logging.INFO)


class FPGMFilterPruner(FilterPruner):
    def __init__(self, model, inputs, sen_file=None, opt=None):
        super(FPGMFilterPruner, self).__init__(
            model, inputs, sen_file=sen_file, opt=opt)

    def cal_mask(self, pruned_ratio, collection):
        var_name = collection.master_name
        pruned_axis = collection.master_axis
        value = collection.values[var_name]
        groups = 1
        for _detail in collection.all_pruning_details():
            assert (isinstance(_detail.axis, int))
            if _detail.axis == 1:
                _groups = _detail.op.attr('groups')
                if _groups is not None and _groups > 1:
                    groups = _groups
                    break

        dist_sum_list = []
        for out_i in range(value.shape[0]):
            dist_sum = self.get_distance_sum(value, out_i)
            dist_sum_list.append(dist_sum)
        scores = np.array(dist_sum_list)

        if groups > 1:
            scores = scores.reshape([groups, -1])
            scores = np.mean(scores, axis=1)

        sorted_idx = scores.argsort()
        pruned_num = int(round(len(sorted_idx) * pruned_ratio))
        pruned_idx = sorted_idx[:pruned_num]
        mask_shape = [value.shape[pruned_axis]]
        mask = np.ones(mask_shape, dtype="int32")
        if groups > 1:
            mask = mask.reshape([groups, -1])
        mask[pruned_idx] = 0
        return mask.reshape(mask_shape)

    def get_distance_sum(self, value, out_idx):
        w = value.view()
        w.shape = value.shape[0], np.product(value.shape[1:])
        selected_filter = np.tile(w[out_idx], (w.shape[0], 1))
        x = w - selected_filter
        x = np.sqrt(np.sum(x * x, -1))
        return x.sum()
