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
from paddlevideo.modeling.framework.estimators.base import BaseEstimator
from paddlevideo.modeling.registry import ESTIMATORS
from paddlevideo.utils import get_logger

from ... import builder

logger = get_logger("paddlevideo")


@ESTIMATORS.register()
class DepthEstimator(BaseEstimator):
    """DepthEstimator
    """
    def forward_net(self, inputs, day_or_night='day_and_night'):
        if self.backbone is not None:
            outputs = self.backbone(inputs, day_or_night)
        else:
            outputs = inputs
        return outputs

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        inputs, _ = data_batch
        outputs = self.forward_net(inputs, day_or_night='day_and_night')
        loss_metrics = self.head.loss(inputs, outputs)
        return loss_metrics

    def val_step(self, data_batch):
        inputs, day_or_night = data_batch
        outputs = self.forward_net(inputs, day_or_night=day_or_night)
        loss_metrics = self.head.loss(inputs, outputs)
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        inputs, day_or_night = data_batch
        outputs = self.forward_net(inputs, day_or_night=day_or_night)
        loss_metrics = self.head.loss(inputs, outputs)
        return loss_metrics

    def infer_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        inputs = data_batch[0]
        outputs = self.forward_net(inputs, day_or_night='day')
        return outputs
