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
from collections import OrderedDict
from paddlevideo.utils import get_logger, load, log_batch, AverageMeter
from .registry import METRIC
from .base import BaseMetric
import time
from datetime import datetime
from .ava_utils import ava_evaluate_results

logger = get_logger("paddlevideo")
""" An example for metrics class.
    MultiCropMetric for slowfast.
"""


@METRIC.register
class AVAMetric(BaseMetric):

    def __init__(self,
                 data_size,
                 batch_size,
                 file_path,
                 exclude_file,
                 label_file,
                 custom_classes,
                 log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)

        self.file_path = file_path
        self.exclude_file = exclude_file
        self.label_file = label_file
        self.custom_classes = custom_classes

        self.results = []

        record_list = [
            ("loss", AverageMeter('loss', '7.5f')),
            ("recall@thr=0.5", AverageMeter("recall@thr=0.5", '.5f')),
            ("prec@thr=0.5", AverageMeter("prec@thr=0.5", '.5f')),
            ("recall@top3", AverageMeter("recall@top3", '.5f')),
            ("prec@top3", AverageMeter("prec@top3", '.5f')),
            ("recall@top5", AverageMeter("recall@top5", '.5f')),
            ("prec@top5", AverageMeter("prec@top5", '.5f')),
            ("mAP@0.5IOU", AverageMeter("mAP@0.5IOU", '.5f')),
            ("batch_time", AverageMeter('batch_cost', '.5f')),
            ("reader_time", AverageMeter('reader_cost', '.5f')),
        ]

        self.record_list = OrderedDict(record_list)

        self.tic = time.time()

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """

        self.results.extend(outputs)
        self.record_list['batch_time'].update(time.time() - self.tic)
        tic = time.time()
        ips = "ips: {:.5f} instance/sec.".format(
            self.batch_size / self.record_list["batch_time"].val)
        log_batch(self.record_list, batch_id, 0, 0, "test", ips)

    def set_dataset_info(self, info, dataset_len):
        self.info = info
        self.dataset_len = dataset_len

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        test_res = ava_evaluate_results(self.info, self.dataset_len,
                                        self.results, None, self.label_file,
                                        self.file_path, self.exclude_file)

        for name, value in test_res.items():
            self.record_list[name].update(value, self.batch_size)

        return self.record_list
