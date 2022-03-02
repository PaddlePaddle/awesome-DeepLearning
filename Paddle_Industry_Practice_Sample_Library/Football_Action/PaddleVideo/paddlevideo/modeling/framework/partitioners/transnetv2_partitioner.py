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

from ...registry import PARTITIONERS
from .base import BasePartitioner

import paddle


@PARTITIONERS.register()
class TransNetV2Partitioner(BasePartitioner):
    """TransNetV2 Partitioner framework
    """
    def forward_net(self, imgs):
        one_hot_pred = self.backbone(imgs)
        return one_hot_pred

    def train_step(self, data_batch):
        """Define how the model is going to train, from input to output.
        """
        frame_sequence = data_batch[0]
        one_hot_gt, many_hot_gt = data_batch[1:]
        one_hot_pred = self.forward_net(frame_sequence)
        dict_ = {}
        if isinstance(one_hot_pred, tuple):
            one_hot_pred, dict_ = one_hot_pred
        many_hot_pred = dict_.get("many_hot", None)
        comb_reg_loss = dict_.get("comb_reg_loss", None)
        loss_metrics = self.head.loss(one_hot_pred, one_hot_gt,
                                    many_hot_pred, many_hot_gt,
                                    reg_losses={"comb_reg": comb_reg_loss})
        return loss_metrics

    def val_step(self, data_batch):
        frame_sequence = data_batch[0]
        one_hot_gt, many_hot_gt = data_batch[1:]
        one_hot_pred = self.forward_net(frame_sequence)
        dict_ = {}
        if isinstance(one_hot_pred, tuple):
            one_hot_pred, dict_ = one_hot_pred
        many_hot_pred = dict_.get("many_hot", None)
        comb_reg_loss = dict_.get("comb_reg_loss", None)
        loss_metrics = self.head.loss(one_hot_pred, one_hot_gt,
                                      many_hot_pred, many_hot_gt,
                                      reg_losses={"comb_reg": comb_reg_loss})
        return loss_metrics

    def test_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        # NOTE: (shipping) when testing, the net won't call head.loss, we deal with the test processing in /paddlevideo/metrics
        frame_sequence = data_batch[0]
        one_hot_pred = self.forward_net(frame_sequence)
        return one_hot_pred

    def infer_step(self, data_batch):
        """Define how the model is going to test, from input to output."""
        frame_sequence = data_batch[0]
        one_hot_pred = self.forward_net(frame_sequence)
        return one_hot_pred
