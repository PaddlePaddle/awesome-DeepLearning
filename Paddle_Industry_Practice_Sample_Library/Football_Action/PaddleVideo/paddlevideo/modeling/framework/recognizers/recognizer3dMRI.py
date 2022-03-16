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

from ...registry import RECOGNIZERS
from .base import BaseRecognizer
from paddlevideo.utils import get_logger
import paddle

logger = get_logger("paddlevideo")


@RECOGNIZERS.register()
class Recognizer3DMRI(BaseRecognizer):
    """3D Recognizer model framework.
    """
    def forward_net(self, imgs):
        """Define how the model is going to run, from input to output.
        """

        imgs[0] = paddle.cast(imgs[0], "float32")
        imgs[1] = paddle.cast(imgs[1], "float32")
        imgs[0] = imgs[0].unsqueeze(1)
        imgs[1] = imgs[1].unsqueeze(1)

        feature = self.backbone(imgs)
        cls_score = self.head(feature)
        return cls_score

    def train_step(self, data_batch):
        """Training step.
        """
        imgs = data_batch[0:2]
        labels = data_batch[2:]

        # call forward
        cls_score = self.forward_net(imgs)
        cls_score = paddle.nn.functional.sigmoid(cls_score)
        loss_metrics = self.head.loss(cls_score, labels, if_top5=False)
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        imgs = data_batch[0:2]
        labels = data_batch[2:]

        # call forward
        cls_score = self.forward_net(imgs)
        cls_score = paddle.nn.functional.sigmoid(cls_score)
        loss_metrics = self.head.loss(cls_score,
                                      labels,
                                      valid_mode=True,
                                      if_top5=False)
        return loss_metrics

    def test_step(self, data_batch):
        """Test step.
        """
        imgs = data_batch[0:2]
        # call forward
        cls_score = self.forward_net(imgs)

        return cls_score

    def infer_step(self, data_batch):
        """Infer step.
        """
        imgs = data_batch[0:2]
        # call forward
        cls_score = self.forward_net(imgs)

        return cls_score
