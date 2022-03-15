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

import cv2
import numpy as np
import paddle.nn as nn
from paddlevideo.utils import get_dist_info
import paddle
from ..builder import build_loss
from ..registry import HEADS

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


@HEADS.register()
class AddsHead(nn.Layer):
    """TimeSformerHead Head.

    Args:
        num_classes (int): The number of classes to be classified.
        in_channels (int): The number of channles in input feature.
        loss_cfg (dict): Config for building config. Default: dict(name='CrossEntropyLoss').
        std(float): Std(Scale) value in normal initilizar. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to initialize.

    """
    def __init__(self,
                 avg_reprojection,
                 disparity_smoothness,
                 no_ssim,
                 loss_cfg=dict(name='ADDSLoss'),
                 max_gt_depth=60,
                 pred_depth_scale_factor=1):

        super(AddsHead, self).__init__()
        loss_cfg['avg_reprojection'] = avg_reprojection
        loss_cfg['disparity_smoothness'] = disparity_smoothness
        loss_cfg['no_ssim'] = no_ssim
        self.max_gt_depth = max_gt_depth
        self.pred_depth_scale_factor = pred_depth_scale_factor
        self.loss_func = build_loss(loss_cfg)

    def forward(self):
        raise NotImplemented

    def loss(self, inputs, outputs):
        if self.training:
            return self.loss_func(inputs, outputs)
        else:
            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.get_metrics(
                outputs['pred_disp'], outputs['gt'])
            outputs['abs_rel'] = abs_rel
            outputs['sq_rel'] = sq_rel
            outputs['rmse'] = rmse
            outputs['rmse_log'] = rmse_log
            outputs['a1'] = a1
            outputs['a2'] = a2
            outputs['a3'] = a3
            return outputs

    def get_metrics(self, pred_disp, gt_depth):
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= self.pred_depth_scale_factor
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        mask2 = gt_depth <= self.max_gt_depth
        pred_depth = pred_depth[mask2]
        gt_depth = gt_depth[mask2]

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = self.compute_errors(
            gt_depth, pred_depth)

        _, world_size = get_dist_info()
        if world_size > 1:
            # educe sum when valid
            # TODO: there are some problems with multi gpu gather code.
            abs_rel = paddle.to_tensor(abs_rel)
            sq_rel = paddle.to_tensor(sq_rel)
            rmse = paddle.to_tensor(rmse)
            rmse_log = paddle.to_tensor(rmse_log)
            a1 = paddle.to_tensor(a1)
            a2 = paddle.to_tensor(a2)
            a3 = paddle.to_tensor(a3)
            abs_rel = paddle.distributed.all_reduce(
                abs_rel, op=paddle.distributed.ReduceOp.SUM) / world_size
            sq_rel = paddle.distributed.all_reduce(
                sq_rel, op=paddle.distributed.ReduceOp.SUM) / world_size
            rmse = paddle.distributed.all_reduce(
                rmse, op=paddle.distributed.ReduceOp.SUM) / world_size
            rmse_log = paddle.distributed.all_reduce(
                rmse_log, op=paddle.distributed.ReduceOp.SUM) / world_size
            a1 = paddle.distributed.all_reduce(
                a1, op=paddle.distributed.ReduceOp.SUM) / world_size
            a2 = paddle.distributed.all_reduce(
                a2, op=paddle.distributed.ReduceOp.SUM) / world_size
            a3 = paddle.distributed.all_reduce(
                a3, op=paddle.distributed.ReduceOp.SUM) / world_size
            return abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(
            ), a1.item(), a2.item(), a3.item()

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

    def compute_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25**2).mean()
        a3 = (thresh < 1.25**3).mean()

        rmse = (gt - pred)**2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred))**2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred)**2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
