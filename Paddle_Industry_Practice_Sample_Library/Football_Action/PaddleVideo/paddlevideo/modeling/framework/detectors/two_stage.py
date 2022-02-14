# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import paddle
import paddle.nn as nn
from ... import builder
import paddle.distributed as dist
from ...registry import DETECTORS
from .base import BaseDetector


@DETECTORS.register()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.  """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = neck  # useless

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = builder.build_head(rpn_head_)

        if roi_head is not None:
            self.roi_head = builder.build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        """whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        """Initialize the weights in detector.  """
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        x = self.backbone(img)
        return x

    def train_step(self, data, **kwargs):
        img_slow = data[0]
        img_fast = data[1]
        proposals, gt_bboxes, gt_labels, scores, entity_ids = self.get_unpad_datas(
            data)
        img_shape = data[7]
        img_idx = data[8]
        img_metas = scores, entity_ids
        x = self.extract_feat(img=[img_slow, img_fast])
        roi_losses = self.roi_head.train_step(x, img_metas, proposals,
                                              gt_bboxes, gt_labels, **kwargs)
        losses = dict()
        losses.update(roi_losses)

        return losses

    def val_step(self, data, rescale=False):
        img_slow = data[0]
        img_fast = data[1]
        proposals, gt_bboxes, gt_labels, scores, entity_ids = self.get_unpad_datas(
            data)
        img_shape = data[7]
        img_metas = scores, entity_ids
        x = self.extract_feat(img=[img_slow, img_fast])

        return self.roi_head.simple_test(x,
                                         proposals[0],
                                         img_shape,
                                         rescale=rescale)

    def test_step(self, data, rescale=False):
        return self.val_step(data, rescale)

    def infer_step(self, data, rescale=False):
        ''' model inference'''

        img_slow = data[0]
        img_fast = data[1]
        proposals = data[2]
        img_shape = data[3]

        # using slowfast model to extract spatio-temporal features
        x = self.extract_feat(img=[img_slow, img_fast])

        ret = self.roi_head.simple_test(x,
                                        proposals[0],
                                        img_shape,
                                        rescale=rescale)
        return ret

    def get_unpad_datas(self, data):
        ''' get original datas padded in dataset '''
        pad_proposals = data[2]
        pad_gt_bboxes = data[3]
        pad_gt_labels = data[4]
        pad_scores, pad_entity_ids = data[5], data[6]
        len_proposals = data[9]
        len_gt_bboxes = data[10]
        len_gt_labels = data[11]
        len_scores = data[12]
        len_entity_ids = data[13]
        N = pad_proposals.shape[0]
        proposals = []
        gt_bboxes = []
        gt_labels = []
        scores = []
        entity_ids = []
        for bi in range(N):
            pad_proposal = pad_proposals[bi]
            len_proposal = len_proposals[bi]
            index_proposal = paddle.arange(len_proposal)
            proposal = paddle.index_select(x=pad_proposal,
                                           index=index_proposal,
                                           axis=0)
            proposals.append(proposal)

            pad_gt_bbox = pad_gt_bboxes[bi]
            len_gt_bbox = len_gt_bboxes[bi]
            index_gt_bbox = paddle.arange(len_gt_bbox)
            gt_bbox = paddle.index_select(x=pad_gt_bbox,
                                          index=index_gt_bbox,
                                          axis=0)
            gt_bboxes.append(gt_bbox)

            pad_gt_label = pad_gt_labels[bi]
            len_gt_label = len_gt_labels[bi]
            index_gt_label = paddle.arange(len_gt_label)
            gt_label = paddle.index_select(x=pad_gt_label,
                                           index=index_gt_label,
                                           axis=0)
            gt_labels.append(gt_label)

            pad_score = pad_scores[bi]
            len_score = len_scores[bi]
            index_score = paddle.arange(len_score)
            score = paddle.index_select(x=pad_score, index=index_score, axis=0)
            scores.append(score)

            pad_entity_id = pad_entity_ids[bi]
            len_entity_id = len_entity_ids[bi]
            index_entity_id = paddle.arange(len_entity_id)
            entity_id = paddle.index_select(x=pad_entity_id,
                                            index=index_entity_id,
                                            axis=0)
            entity_ids.append(entity_id)

        return proposals, gt_bboxes, gt_labels, scores, entity_ids
