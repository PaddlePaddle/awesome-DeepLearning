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

from abc import abstractmethod

import paddle
import paddle.nn as nn
from paddlevideo.modeling.registry import ESTIMATORS
from paddlevideo.utils import get_logger

from ... import builder

logger = get_logger("paddlevideo")


@ESTIMATORS.register()
class BaseEstimator(nn.Layer):
    """BaseEstimator

    """
    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
            if hasattr(self.backbone, 'init_weights'):
                self.backbone.init_weights()
        else:
            self.backbone = None

        if head is not None:
            self.head_name = head.name
            self.head = builder.build_head(head)
            if hasattr(self.head, 'init_weights'):
                self.head.init_weights()
        else:
            self.head = None

    def forward(self, data_batch, mode='infer'):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        """
        if mode == 'train':
            return self.train_step(data_batch)
        elif mode == 'valid':
            return self.val_step(data_batch)
        elif mode == 'test':
            return self.test_step(data_batch)
        elif mode == 'infer':
            return self.infer_step(data_batch)
        else:
            raise NotImplementedError

    @abstractmethod
    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data_batch):
        """Define how the model is going to valid, from input to output."""
        raise NotImplementedError

    @abstractmethod
    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        raise NotImplementedError

    @abstractmethod
    def infer_step(self, data_batch):
        """Define how the model is going to infer, from input to output."""
        raise NotImplementedError
