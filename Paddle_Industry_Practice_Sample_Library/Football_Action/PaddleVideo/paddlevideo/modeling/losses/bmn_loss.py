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
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class BMNLoss(BaseWeightedLoss):
    """Loss for BMN model
    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
    """
    def __init__(self, dscale, tscale, datatype='float32'):
        super().__init__()
        self.dscale = dscale
        self.tscale = tscale
        self.datatype = datatype

    def _get_mask(self, dscale, tscale):
        bm_mask = []
        for idx in range(dscale):
            mask_vector = [1 for i in range(tscale - idx)
                           ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        bm_mask = paddle.to_tensor(bm_mask)
        bm_mask.stop_gradient = True
        return bm_mask

    def tem_loss_func(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label, datatype):
            pred_score = paddle.reshape(x=pred_score, shape=[-1])
            gt_label = paddle.reshape(x=gt_label, shape=[-1])
            gt_label.stop_gradient = True
            pmask = paddle.cast(x=(gt_label > 0.5), dtype=datatype)
            num_entries = paddle.cast(paddle.shape(pmask), dtype=datatype)
            num_positive = paddle.cast(paddle.sum(pmask), dtype=datatype)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            temp = paddle.log(pred_score + epsilon)
            loss_pos = paddle.multiply(paddle.log(pred_score + epsilon), pmask)
            loss_pos = coef_1 * paddle.mean(loss_pos)
            loss_neg = paddle.multiply(paddle.log(1.0 - pred_score + epsilon),
                                       (1.0 - pmask))
            loss_neg = coef_0 * paddle.mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start, self.datatype)
        loss_end = bi_loss(pred_end, gt_end, self.datatype)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(self, pred_score, gt_iou_map, mask):

        gt_iou_map = paddle.multiply(gt_iou_map, mask)

        u_hmask = paddle.cast(x=gt_iou_map > 0.7, dtype=self.datatype)
        u_mmask = paddle.logical_and(gt_iou_map <= 0.7, gt_iou_map > 0.3)
        u_mmask = paddle.cast(x=u_mmask, dtype=self.datatype)
        u_lmask = paddle.logical_and(gt_iou_map <= 0.3, gt_iou_map >= 0.)
        u_lmask = paddle.cast(x=u_lmask, dtype=self.datatype)
        u_lmask = paddle.multiply(u_lmask, mask)

        num_h = paddle.cast(paddle.sum(u_hmask), dtype=self.datatype)
        num_m = paddle.cast(paddle.sum(u_mmask), dtype=self.datatype)
        num_l = paddle.cast(paddle.sum(u_lmask), dtype=self.datatype)

        r_m = num_h / num_m
        u_smmask = paddle.uniform(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=self.datatype,
            min=0.0,
            max=1.0)
        u_smmask = paddle.multiply(u_mmask, u_smmask)
        u_smmask = paddle.cast(x=(u_smmask > (1. - r_m)), dtype=self.datatype)

        r_l = num_h / num_l
        u_slmask = paddle.uniform(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=self.datatype,
            min=0.0,
            max=1.0)
        u_slmask = paddle.multiply(u_lmask, u_slmask)
        u_slmask = paddle.cast(x=(u_slmask > (1. - r_l)), dtype=self.datatype)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True
        loss = F.square_error_cost(pred_score, gt_iou_map)
        loss = paddle.multiply(loss, weights)
        loss = 0.5 * paddle.sum(loss) / paddle.sum(weights)

        return loss

    def pem_cls_loss_func(self, pred_score, gt_iou_map, mask):
        gt_iou_map = paddle.multiply(gt_iou_map, mask)
        gt_iou_map.stop_gradient = True
        pmask = paddle.cast(x=(gt_iou_map > 0.9), dtype=self.datatype)
        nmask = paddle.cast(x=(gt_iou_map <= 0.9), dtype=self.datatype)
        nmask = paddle.multiply(nmask, mask)

        num_positive = paddle.sum(pmask)
        num_entries = num_positive + paddle.sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = paddle.multiply(paddle.log(pred_score + epsilon), pmask)
        loss_pos = coef_1 * paddle.sum(loss_pos)
        loss_neg = paddle.multiply(paddle.log(1.0 - pred_score + epsilon),
                                   nmask)
        loss_neg = coef_0 * paddle.sum(loss_neg)
        loss = -1 * (loss_pos + loss_neg) / num_entries
        return loss

    def forward(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start,
                gt_end):
        pred_bm_reg = paddle.squeeze(paddle.slice(pred_bm,
                                                  axes=[1],
                                                  starts=[0],
                                                  ends=[1]),
                                     axis=[1])
        pred_bm_cls = paddle.squeeze(paddle.slice(pred_bm,
                                                  axes=[1],
                                                  starts=[1],
                                                  ends=[2]),
                                     axis=[1])

        bm_mask = self._get_mask(self.dscale, self.tscale)

        pem_reg_loss = self.pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

        tem_loss = self.tem_loss_func(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
        return loss
