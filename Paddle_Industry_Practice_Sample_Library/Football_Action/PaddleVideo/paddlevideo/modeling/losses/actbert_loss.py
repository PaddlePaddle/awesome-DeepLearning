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
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class ActBertLoss(BaseWeightedLoss):
    """Loss for ActBert model
    """
    def __init__(self, vocab_size=30522, a_target_size=700):
        super().__init__()
        self.vocab_size = vocab_size
        self.a_target_size = a_target_size
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.vis_criterion = nn.KLDivLoss(reduction="none")

    def forward(self, prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score, \
                text_labels, image_label, image_target, action_label, next_sentence_label):
        """
        Args:
            text_label: text label(with mask). Shape: [batch_size, seqence_length]
            image_label: image label(with mask). Shape: [batch_size, region_length]
            image_target: label of image feature distribution,
                            Shape: [batch_size, region_length-1, num_image_class](minus 1 for xxx).
            action label: action label(with mask), Shape: [batch_size, action_length]
            next_sentence_label: is next sentence or not. Shape: [batch_size]
        """
        prediction_scores_v = prediction_scores_v[:,
                                                  1:]  #8,37,1601 --> 8,36,1601

        img_loss = self.vis_criterion(
            F.log_softmax(prediction_scores_v, axis=2),
            image_target  #8,36,1601
        )
        masked_img_loss = paddle.sum(
            img_loss * (image_label == 1).unsqueeze(2).astype('float32')) / max(
                paddle.sum((image_label == 1).astype('float32')), 1e-6)

        masked_text_loss = self.loss_fct(
            prediction_scores_t.reshape([-1, self.vocab_size]),  #8,36,30522
            text_labels.reshape([-1]),  #8,36   # label -1 will be ignored
        )

        masked_action_loss = self.loss_fct(
            prediction_scores_a.reshape([-1, self.a_target_size]),  #8,5,700
            action_label.reshape([-1]),  #8,5
        )

        next_sentence_loss = self.loss_fct(
            seq_relationship_score.reshape([-1, 2]),
            next_sentence_label.reshape([-1])  #8,2
        )

        total_loss = masked_text_loss.unsqueeze(0) + masked_img_loss.unsqueeze(
            0) + masked_action_loss.unsqueeze(0) + next_sentence_loss.unsqueeze(
                0)
        return total_loss
