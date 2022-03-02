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

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class RecognizerGCN(BaseRecognizer):
    """GCN Recognizer model framework.
    """
    def forward_net(self, data):
        """Define how the model is going to run, from input to output.
        """
        feature = self.backbone(data)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        data = data_batch[0]
        label = data_batch[1:]

        # call forward
        cls_score = self.forward_net(data)
        loss_metrics = self.head.loss(cls_score, label)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        data = data_batch[0]
        label = data_batch[1:]

        # call forward
        cls_score = self.forward_net(data)
        loss_metrics = self.head.loss(cls_score, label, valid_mode=True)
        return loss_metrics

    def test_step(self, data_batch):
        """Test step.
        """
        data = data_batch[0]

        # call forward
        cls_score = self.forward_net(data)
        return cls_score

    def infer_step(self, data_batch):
        """Infer step.
        """
        data = data_batch[0]

        # call forward
        cls_score = self.forward_net(data)
        return cls_score
