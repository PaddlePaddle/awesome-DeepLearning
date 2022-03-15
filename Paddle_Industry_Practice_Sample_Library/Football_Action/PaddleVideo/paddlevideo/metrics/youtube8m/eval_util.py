# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides functions to help with evaluating models."""
import numpy as np
import paddle
from paddlevideo.utils import get_logger

from ..base import BaseMetric
from ..registry import METRIC
from . import average_precision_calculator as ap_calculator
from . import mean_average_precision_calculator as map_calculator

logger = get_logger("paddlevideo")


def flatten(l):
    """ Merges a list of lists into a single list. """
    return [item for sublist in l for item in sublist]


def calculate_hit_at_one(predictions, actuals):
    """Performs a local (numpy) calculation of the hit at one.

    Args:
        predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
        actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.

    Returns:
        float: The average hit at one across the entire batch.
    """
    top_prediction = np.argmax(predictions, 1)
    hits = actuals[np.arange(actuals.shape[0]), top_prediction]
    return np.mean(hits)


def calculate_precision_at_equal_recall_rate(predictions, actuals):
    """Performs a local (numpy) calculation of the PERR.

    Args:
        predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
        actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.

    Returns:
        float: The average precision at equal recall rate across the entire batch.
    """
    aggregated_precision = 0.0
    num_videos = actuals.shape[0]
    for row in np.arange(num_videos):
        num_labels = int(np.sum(actuals[row]))
        top_indices = np.argpartition(predictions[row],
                                      -num_labels)[-num_labels:]
        item_precision = 0.0
        for label_index in top_indices:
            if predictions[row][label_index] > 0:
                item_precision += actuals[row][label_index]
        item_precision /= top_indices.size
        aggregated_precision += item_precision
    aggregated_precision /= num_videos
    return aggregated_precision


def calculate_gap(predictions, actuals, top_k=20):
    """Performs a local (numpy) calculation of the global average precision.

    Only the top_k predictions are taken for each of the videos.

    Args:
        predictions: Matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
        actuals: Matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
        top_k: How many predictions to use per video.

    Returns:
        float: The global average precision.
    """
    gap_calculator = ap_calculator.AveragePrecisionCalculator()
    sparse_predictions, sparse_labels, num_positives = top_k_by_class(
        predictions, actuals, top_k)
    gap_calculator.accumulate(flatten(sparse_predictions),
                              flatten(sparse_labels), sum(num_positives))
    return gap_calculator.peek_ap_at_n()


def top_k_by_class(predictions, labels, k=20):
    """Extracts the top k predictions for each video, sorted by class.

    Args:
        predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
        k: the top k non-zero entries to preserve in each prediction.

    Returns:
        A tuple (predictions,labels, true_positives). 'predictions' and 'labels'
        are lists of lists of floats. 'true_positives' is a list of scalars. The
        length of the lists are equal to the number of classes. The entries in the
        predictions variable are probability predictions, and
        the corresponding entries in the labels variable are the ground truth for
        those predictions. The entries in 'true_positives' are the number of true
        positives for each class in the ground truth.

    Raises:
        ValueError: An error occurred when the k is not a positive integer.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    k = min(k, predictions.shape[1])
    num_classes = predictions.shape[1]
    prediction_triplets = []
    for video_index in range(predictions.shape[0]):
        prediction_triplets.extend(
            top_k_triplets(predictions[video_index], labels[video_index], k))
    out_predictions = [[] for v in range(num_classes)]
    out_labels = [[] for v in range(num_classes)]
    for triplet in prediction_triplets:
        out_predictions[triplet[0]].append(triplet[1])
        out_labels[triplet[0]].append(triplet[2])
    out_true_positives = [np.sum(labels[:, i]) for i in range(num_classes)]

    return out_predictions, out_labels, out_true_positives


def top_k_triplets(predictions, labels, k=20):
    """Get the top_k for a 1-d numpy array. Returns a sparse list of tuples in
    (prediction, class) format"""
    m = len(predictions)
    k = min(k, m)
    indices = np.argpartition(predictions, -k)[-k:]
    return [(index, predictions[index], labels[index]) for index in indices]


@METRIC.register
class HitOneMetric(BaseMetric):
    """A class to store the evaluation metrics."""
    def __init__(self,
                 num_class,
                 top_k,
                 data_size,
                 batch_size,
                 log_interval=20):
        """Construct an HitOneMetric object to store the evaluation metrics."""
        self.hit_at_one = []
        self.perr = []
        self.gap = []
        super().__init__(data_size, batch_size, log_interval)

    def accumulate(self):
        logger.info(
            '[TEST] finished, hit_at_one = {:.5f}, perr = {:.5f}, gap = {:.5f}'.
            format(np.mean(np.array(self.hit_at_one)),
                   np.mean(np.array(self.perr)), np.mean(np.array(self.gap))))

    def clear(self):
        """Clear the evaluation metrics and reset the HitOneMetric object."""
        self.hit_at_one = []
        self.perr = []
        self.gap = []

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        hit_at_one = paddle.to_tensor(outputs['hit_at_one'])
        perr = paddle.to_tensor(outputs['perr'])
        gap = paddle.to_tensor(outputs['gap'])
        # NOTE(shipping): deal with multi cards validate
        if self.world_size > 1:
            hit_at_one = paddle.distributed.all_reduce(
                hit_at_one,
                op=paddle.distributed.ReduceOp.SUM) / self.world_size
            perr = paddle.distributed.all_reduce(
                perr, op=paddle.distributed.ReduceOp.SUM) / self.world_size
            gap = paddle.distributed.all_reduce(
                gap, op=paddle.distributed.ReduceOp.SUM) / self.world_size

        self.hit_at_one.append(hit_at_one.numpy())
        self.perr.append(perr.numpy())
        self.gap.append(gap.numpy())
        # preds ensemble
        if batch_id % self.log_interval == 0:
            logger.info("[TEST] Processing batch {}/{}...".format(
                batch_id,
                self.data_size // (self.batch_size * self.world_size),
            ))
