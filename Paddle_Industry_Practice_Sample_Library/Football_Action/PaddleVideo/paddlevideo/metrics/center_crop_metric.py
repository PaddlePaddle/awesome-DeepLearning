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

import numpy as np
import paddle
from paddle.hapi.model import _all_gather

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger
logger = get_logger("paddlevideo")


@METRIC.register
class CenterCropMetric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.top1 = []
        self.top5 = []

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        labels = data[1]

        top1 = paddle.metric.accuracy(input=outputs, label=labels, k=1)
        top5 = paddle.metric.accuracy(input=outputs, label=labels, k=5)
        #NOTE(shipping): deal with multi cards validate
        if self.world_size > 1:
            top1 = paddle.distributed.all_reduce(
                top1, op=paddle.distributed.ReduceOp.SUM) / self.world_size
            top5 = paddle.distributed.all_reduce(
                top5, op=paddle.distributed.ReduceOp.SUM) / self.world_size

        self.top1.append(top1.numpy())
        self.top5.append(top5.numpy())
        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        logger.info('[TEST] finished, avg_acc1= {}, avg_acc5= {} '.format(
            np.mean(np.array(self.top1)), np.mean(np.array(self.top5))))
