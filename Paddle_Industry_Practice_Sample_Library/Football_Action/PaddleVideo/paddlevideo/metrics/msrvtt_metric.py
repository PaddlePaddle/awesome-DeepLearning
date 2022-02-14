# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.hapi.model import _all_gather

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@METRIC.register
class MSRVTTMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.score_matrix = np.zeros((data_size, data_size))
        self.target_matrix = np.zeros((data_size, data_size))
        self.rank_matrix = np.ones((data_size)) * data_size

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        target = data[-1]
        cm_logit = outputs[-1]

        self.score_matrix[batch_id, :] = F.softmax(
            cm_logit, axis=1)[:, 0].reshape([-1]).numpy()
        self.target_matrix[batch_id, :] = target.reshape([-1]).numpy()

        rank = np.where((np.argsort(-self.score_matrix[batch_id]) == np.where(
            self.target_matrix[batch_id] == 1)[0][0]) == 1)[0][0]
        self.rank_matrix[batch_id] = rank

        rank_matrix_tmp = self.rank_matrix[:batch_id + 1]
        r1 = 100.0 * np.sum(rank_matrix_tmp < 1) / len(rank_matrix_tmp)
        r5 = 100.0 * np.sum(rank_matrix_tmp < 5) / len(rank_matrix_tmp)
        r10 = 100.0 * np.sum(rank_matrix_tmp < 10) / len(rank_matrix_tmp)

        medr = np.floor(np.median(rank_matrix_tmp) + 1)
        meanr = np.mean(rank_matrix_tmp) + 1
        logger.info(
            "[{}] Final r1:{:.3f}, r5:{:.3f}, r10:{:.3f}, mder:{:.3f}, meanr:{:.3f}"
            .format(batch_id, r1, r5, r10, medr, meanr))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        logger.info("Eval Finished!")
