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

from .registry import METRIC
from .base import BaseMetric
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


def predictions_to_scenes(predictions):
    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)


def evaluate_scenes(gt_scenes, pred_scenes, n_frames_miss_tolerance=2):
    """
    Adapted from: https://github.com/gyglim/shot-detection-evaluation
    The original based on: http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=19

    n_frames_miss_tolerance:
        Number of frames it is possible to miss ground truth by, and still being counted as a correct detection.

    Examples of computation with different tolerance margin:
    n_frames_miss_tolerance = 0
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.5, 5.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.5, 5.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.5, 4.5]] -> MISS
    n_frames_miss_tolerance = 1
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[5.0, 6.0]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[5.0, 6.0]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[4.0, 5.0]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[3.0, 4.0]] -> MISS
    n_frames_miss_tolerance = 2
      pred_scenes: [[0, 5], [6, 9]] -> pred_trans: [[4.5, 6.5]]
      gt_scenes:   [[0, 5], [6, 9]] -> gt_trans:   [[4.5, 6.5]] -> HIT
      gt_scenes:   [[0, 4], [5, 9]] -> gt_trans:   [[3.5, 5.5]] -> HIT
      gt_scenes:   [[0, 3], [4, 9]] -> gt_trans:   [[2.5, 4.5]] -> HIT
      gt_scenes:   [[0, 2], [3, 9]] -> gt_trans:   [[1.5, 3.5]] -> MISS

      Users should be careful about adopting these functions in any commercial matters.
    """

    shift = n_frames_miss_tolerance / 2
    gt_scenes = gt_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])
    pred_scenes = pred_scenes.astype(np.float32) + np.array([[-0.5 + shift, 0.5 - shift]])

    gt_trans = np.stack([gt_scenes[:-1, 1], gt_scenes[1:, 0]], 1)
    pred_trans = np.stack([pred_scenes[:-1, 1], pred_scenes[1:, 0]], 1)

    i, j = 0, 0
    tp, fp, fn = 0, 0, 0

    while i < len(gt_trans) or j < len(pred_trans):
        if j == len(pred_trans) or pred_trans[j, 0] > gt_trans[i, 1]:
            fn += 1
            i += 1
        elif i == len(gt_trans) or pred_trans[j, 1] < gt_trans[i, 0]:
            fp += 1
            j += 1
        else:
            i += 1
            j += 1
            tp += 1

    if tp + fp != 0:
        p = tp / (tp + fp)
    else:
        p = 0

    if tp + fn != 0:
        r = tp / (tp + fn)
    else:
        r = 0

    if p + r != 0:
        f1 = (p * r * 2) / (p + r)
    else:
        f1 = 0

    assert tp + fn == len(gt_trans)
    assert tp + fp == len(pred_trans)

    return p, r, f1, (tp, fp, fn)


def create_scene_based_summaries(one_hot_pred, one_hot_gt):
    thresholds = np.array([
        0.02, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    ])
    precision, recall, f1, tp, fp, fn = np.zeros_like(thresholds), np.zeros_like(thresholds),\
                                        np.zeros_like(thresholds), np.zeros_like(thresholds),\
                                        np.zeros_like(thresholds), np.zeros_like(thresholds)

    gt_scenes = predictions_to_scenes(one_hot_gt)
    for i in range(len(thresholds)):
        pred_scenes = predictions_to_scenes(
            (one_hot_pred > thresholds[i]).astype(np.uint8)
        )
        precision[i], recall[i], f1[i], (tp[i], fp[i], fn[i]) = evaluate_scenes(gt_scenes, pred_scenes)

    best_idx = np.argmax(f1)

    return f1[best_idx]


@METRIC.register
class TransNetV2Metric(BaseMetric):
    def __init__(self, data_size, batch_size, log_interval=1):
        """prepare for metrics
        """
        super().__init__(data_size, batch_size, log_interval)
        self.predictions = []
        self.total_stats = {"tp": 0, "fp": 0, "fn": 0}

    def update(self, batch_id, data, one_hot):
        """update metrics during each iter
        """
        if isinstance(one_hot, tuple):
            one_hot = one_hot[0]
        one_hot = paddle.nn.functional.sigmoid(one_hot)[0]
        self.predictions.append(one_hot.numpy()[25:75])
        gt_scenes = data[1]
        is_new_file = data[2]
        if is_new_file:
            self.compute(gt_scenes)
        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{} ...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size)))

    def compute(self, gt_scenes):
        predictions = np.concatenate(self.predictions, 0)[:len(frames)]
        _, _, _, (tp, fp, fn), fp_mistakes, fn_mistakes = evaluate_scenes(
            gt_scenes, predictions_to_scenes((predictions >= args.thr).astype(np.uint8)))

        self.total_stats["tp"] += tp
        self.total_stats["fp"] += fp
        self.total_stats["fn"] += fn

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        p = self.total_stats["tp"] / (self.total_stats["tp"] + self.total_stats["fp"])
        r = self.total_stats["tp"] / (self.total_stats["tp"] + self.total_stats["fn"])
        f1 = (p * r * 2) / (p + r)
        logger.info('[TEST] finished, Precision= {:5.2f}, Recall= {:5.2f} , F1 Score= {:5.2f} '.format(
            p * 100, r * 100, f1 * 100))