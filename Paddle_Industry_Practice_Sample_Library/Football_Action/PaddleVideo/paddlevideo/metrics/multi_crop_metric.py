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

from paddlevideo.utils import get_logger
from .registry import METRIC
from .base import BaseMetric

logger = get_logger("paddlevideo")
""" An example for metrics class.
    MultiCropMetric for slowfast.
"""


@METRIC.register
class MultiCropMetric(BaseMetric):
    def __init__(self,
                 data_size,
                 batch_size,
                 num_ensemble_views,
                 num_spatial_crops,
                 num_classes,
                 log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.num_ensemble_views = num_ensemble_views
        self.num_spatial_crops = num_spatial_crops
        self.num_classes = num_classes

        self.num_clips = self.num_ensemble_views * self.num_spatial_crops
        num_videos = self.data_size // self.num_clips
        self.video_preds = np.zeros((num_videos, self.num_classes))
        self.video_labels = np.zeros((num_videos, 1), dtype="int64")
        self.clip_count = {}

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        labels = data[2]
        clip_ids = data[3]

        # gather mulit card, results of following process in each card is the same.
        if self.world_size > 1:
            outputs = _all_gather(outputs, self.world_size)
            labels = _all_gather(labels, self.world_size)
            clip_ids = _all_gather(clip_ids, self.world_size)

        # to numpy
        preds = outputs.numpy()
        labels = labels.numpy().astype("int64")
        clip_ids = clip_ids.numpy()

        # preds ensemble
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            ts_idx = int(clip_ids[ind]) % self.num_clips
            if vid_id not in self.clip_count:
                self.clip_count[vid_id] = []
            if ts_idx in self.clip_count[vid_id]:
                logger.info(
                    "[TEST] Passed!! read video {} clip index {} / {} repeatedly."
                    .format(vid_id, ts_idx, clip_ids[ind]))
            else:
                self.clip_count[vid_id].append(ts_idx)
                self.video_preds[vid_id] += preds[ind]  # ensemble method: sum
                if self.video_labels[vid_id].sum() > 0:
                    assert self.video_labels[vid_id] == labels[ind]
                self.video_labels[vid_id] = labels[ind]
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        # check clip index of each video
        for key in self.clip_count.keys():
            if len(self.clip_count[key]) != self.num_clips or sum(
                    self.clip_count[key]) != self.num_clips * (self.num_clips -
                                                               1) / 2:
                logger.info(
                    "[TEST] Count Error!! video [{}] clip count [{}] not match number clips {}"
                    .format(key, self.clip_count[key], self.num_clips))

        video_preds = paddle.to_tensor(self.video_preds)
        video_labels = paddle.to_tensor(self.video_labels)
        acc_top1 = paddle.metric.accuracy(input=video_preds,
                                          label=video_labels,
                                          k=1)
        acc_top5 = paddle.metric.accuracy(input=video_preds,
                                          label=video_labels,
                                          k=5)
        logger.info('[TEST] finished, avg_acc1= {}, avg_acc5= {} '.format(
            acc_top1.numpy(), acc_top5.numpy()))
