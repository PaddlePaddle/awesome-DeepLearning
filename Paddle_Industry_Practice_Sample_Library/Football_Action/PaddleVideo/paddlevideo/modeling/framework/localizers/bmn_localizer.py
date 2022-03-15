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

from ...registry import LOCALIZERS
from .base import BaseLocalizer

import paddle


@LOCALIZERS.register()
class BMNLocalizer(BaseLocalizer):
    """BMN Localization framework
    """
    def forward_net(self, imgs):
        """Call backbone forward.
        """
        preds = self.backbone(imgs)
        return preds

    def train_step(self, data_batch):
        """Training step.
        """
        x_data = data_batch[0]
        gt_iou_map = data_batch[1]
        gt_start = data_batch[2]
        gt_end = data_batch[3]
        gt_iou_map.stop_gradient = True
        gt_start.stop_gradient = True
        gt_end.stop_gradient = True

        # call Model forward
        pred_bm, pred_start, pred_end = self.forward_net(x_data)
        # call Loss forward
        loss = self.loss(pred_bm, pred_start, pred_end, gt_iou_map, gt_start,
                         gt_end)
        avg_loss = paddle.mean(loss)
        loss_metrics = dict()
        loss_metrics['loss'] = avg_loss
        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        return self.train_step(data_batch)

    def test_step(self, data_batch):
        """Test step.
        """
        x_data = data_batch[0]
        pred_bm, pred_start, pred_end = self.forward_net(x_data)
        return pred_bm, pred_start, pred_end

    def infer_step(self, data_batch):
        """Infer step
        """
        x_data = data_batch[0]

        # call Model forward
        pred_bm, pred_start, pred_end = self.forward_net(x_data)
        return pred_bm, pred_start, pred_end
