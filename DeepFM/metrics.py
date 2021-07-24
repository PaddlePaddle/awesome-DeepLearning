# -*- encoding:utf-8 -*-
import numpy as np
'''
     工具类， 用于计算 预测结果评估的基尼系数。
'''
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
