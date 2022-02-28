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

import numpy as np
import paddle
import paddle.nn as nn
from .. import builder
from ..registry import HEADS


def bbox2result(bboxes, labels, num_classes, img_shape, thr=0.01):
    """Convert detection results to a list of numpy arrays.  """
    if len(bboxes) == 0:
        return list(np.zeros((num_classes - 1, 0, 5), dtype=np.float32))
    else:
        bboxes = bboxes[0]
        labels = labels
        img_shape_np = img_shape
        img_h, img_w = img_shape_np[0][0], img_shape_np[0][1]

        img_w = paddle.cast(img_w, dtype='int32')
        img_h = paddle.cast(img_h, dtype='int32')

        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        # We only handle multilabel now
        assert labels.shape[-1] > 1

        scores = labels  # rename
        thr = (thr, ) * num_classes if isinstance(thr, float) else thr
        assert scores.shape[1] == num_classes
        assert len(thr) == num_classes

        result = []
        for i in range(num_classes - 1):
            #step1. 对该类, 每个bbox的得分是否大于阈值
            where = scores[:, i + 1] > thr[i + 1]

            where = paddle.nonzero(where)  # index
            bboxes_select = paddle.index_select(x=bboxes, index=where)
            bboxes_select = bboxes_select[:, :4]

            scores_select = paddle.index_select(x=scores, index=where)
            scores_select = scores_select[:, i + 1:i + 2]

            result.append(
                #对于step1中得分大于阈值的bbox(可能为空), 将bbox及在该类的score放入result列表.
                paddle.concat((bboxes_select, scores_select), axis=1))

        return result


@HEADS.register()
class AVARoIHead(nn.Layer):

    def __init__(self,
                 assigner,
                 sampler,
                 pos_weight=1.0,
                 action_thr=0.0,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.assigner = assigner
        self.sampler = sampler
        self.pos_weight = pos_weight
        self.action_thr = action_thr
        self.init_assigner_sampler()
        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        self.bbox_assigner = builder.build_assigner(self.assigner)
        self.bbox_sampler = builder.build_sampler(self.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = builder.build_roi_extractor(
            bbox_roi_extractor)
        self.bbox_head = builder.build_head(bbox_head)

    def _bbox_forward(self, x, rois, rois_num):
        bbox_feat = self.bbox_roi_extractor(x, rois, rois_num)
        cls_score, bbox_pred = self.bbox_head(
            bbox_feat, rois, rois_num
        )  #deal with: when roi's width or height = 0 , roi_align is wrong
        bbox_results = dict(cls_score=cls_score,
                            bbox_pred=bbox_pred,
                            bbox_feats=bbox_feat)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        rois = [res.bboxes for res in sampling_results]
        rois_num = [res.bboxes.shape[0] for res in sampling_results]
        bbox_results = self._bbox_forward(x, rois, rois_num)
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.pos_weight)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_targets)
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def train_step(self, x, img_metas, proposal_list, gt_bboxes, gt_labels):
        #1. assign gts and sample proposals
        num_imgs = len(img_metas[0])
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(proposal_list[i],
                                                      gt_bboxes[i],
                                                      gt_labels[i])
            sampling_result = self.bbox_sampler.sample(assign_result,
                                                       proposal_list[i],
                                                       gt_bboxes[i],
                                                       gt_labels[i])
            sampling_results.append(sampling_result)

        #2. forward and loss
        bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes,
                                                gt_labels)
        losses = dict()
        losses.update(bbox_results['loss_bbox'])

        return losses

    def simple_test(self, x, proposal_list, img_shape, rescale=False):
        x_shape = x[0].shape
        #assert x_shape[0] == 1, 'only accept 1 sample at test mode'

        det_bboxes, det_labels = self.simple_test_bboxes(x,
                                                         img_shape,
                                                         proposal_list,
                                                         self.action_thr,
                                                         rescale=rescale)

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes, img_shape,
                                   self.action_thr)
        return [bbox_results]

    def simple_test_bboxes(self,
                           x,
                           img_shape,
                           proposals,
                           action_thr,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = [proposals]
        rois_num = [rois[0].shape[0]]
        bbox_results = self._bbox_forward(x, rois, rois_num)
        cls_score = bbox_results['cls_score']
        crop_quadruple = np.array([0, 0, 1, 1])
        flip = False
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            img_shape,
            flip=flip,
            crop_quadruple=crop_quadruple)

        return det_bboxes, det_labels
