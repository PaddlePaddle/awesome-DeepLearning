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

import paddle
import paddle.nn.functional as F
from paddlevideo.utils import get_logger

from ...registry import RECOGNIZERS
from .base import BaseRecognizer

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class RecognizerTransformer_MRI(BaseRecognizer):
    """Transformer's recognizer model framework."""
    def forward_net(self, imgs):
        # imgs.shape=[N,C,T,H,W], for transformer case

        imgs = paddle.cast(imgs, "float32")  #############
        imgs = imgs.unsqueeze(1)

        if self.backbone != None:
            feature = self.backbone(imgs)
        else:
            feature = imgs

        if self.head != None:
            cls_score = self.head(feature)
        else:
            cls_score = None

        return cls_score

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)
        cls_score = paddle.nn.functional.sigmoid(cls_score)
        loss_metrics = self.head.loss(cls_score, labels, if_top5=False)
        return loss_metrics

    def val_step(self, data_batch):
        imgs = data_batch[0]
        labels = data_batch[1:]
        cls_score = self.forward_net(imgs)
        cls_score = paddle.nn.functional.sigmoid(cls_score)
        loss_metrics = self.head.loss(cls_score,
                                      labels,
                                      valid_mode=True,
                                      if_top5=False)
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        num_views = imgs.shape[2] // self.backbone.seg_num
        cls_score = []
        for i in range(num_views):
            view = imgs[:, :, i * self.backbone.seg_num:(i + 1) *
                        self.backbone.seg_num]
            cls_score.append(self.forward_net(view))
        cls_score = self.average_view(cls_score)
        return cls_score

    def infer_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        imgs = data_batch[0]
        num_views = imgs.shape[2] // self.backbone.seg_num
        cls_score = []
        for i in range(num_views):
            view = imgs[:, :, i * self.backbone.seg_num:(i + 1) *
                        self.backbone.seg_num]
            cls_score.append(self.forward_net(view))
        cls_score = self.average_view(cls_score)
        return cls_score

    def average_view(self, cls_score, average_type='score'):
        """Combine the scores of different views

        Args:
            cls_score (list): Scores of multiple views
            average_type (str, optional): Average calculation method. Defaults to 'score'.
        """
        assert average_type in ['score', 'prob'], \
            f"Currently only the average of 'score' or 'prob' is supported, but got {average_type}"
        if average_type == 'score':
            return paddle.add_n(cls_score) / len(cls_score)
        elif average_type == 'avg':
            return paddle.add_n([F.softmax(score)
                                 for score in cls_score]) / len(cls_score)
        else:
            raise NotImplementedError
