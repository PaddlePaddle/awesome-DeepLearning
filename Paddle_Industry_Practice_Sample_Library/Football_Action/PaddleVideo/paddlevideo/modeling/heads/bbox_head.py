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
import paddle.nn.functional as F
import numpy as np
from .. import builder

from ..registry import HEADS

@HEADS.register()
class BBoxHeadAVA(nn.Layer):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively.  """

    def __init__(
            self,
            temporal_pool_type='avg',
            spatial_pool_type='max',
            in_channels=2048,
            num_classes=81,# The first class is reserved, to classify bbox as pos / neg
            dropout_ratio=0,
            dropout_before_pool=True,
            topk=(3, 5),
            multilabel=True):

        super(BBoxHeadAVA, self).__init__()
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool

        self.multilabel = multilabel
        if topk is None:
            self.topk = ()
        elif isinstance(topk, int):
            self.topk = (topk, )
        elif isinstance(topk, tuple):
            assert all([isinstance(k, int) for k in topk])
            self.topk = topk
        else:
            raise TypeError('topk should be int or tuple[int], '
                            f'but get {type(topk)}')
        # Class 0 is ignored when calculaing multilabel accuracy,
        # so topk cannot be equal to num_classes
        assert all([k < num_classes for k in self.topk])
        assert self.multilabel

        in_channels = self.in_channels
        if self.temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3D((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3D((1, None, None))
        if self.spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3D((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3D((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        weight_attr = paddle.framework.ParamAttr(name="weight",
                                                 initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.01))
        bias_attr = paddle.ParamAttr(name="bias",
                                     initializer=paddle.nn.initializer.Constant(value=0.0))

        self.fc_cls = nn.Linear(in_channels, num_classes, weight_attr=weight_attr, bias_attr=bias_attr)

        self.debug_imgs = None

    def forward(self, x,rois, rois_num):
        roi = paddle.concat(rois)
        roi_x1 = paddle.index_select(roi, index=paddle.to_tensor(0), axis=1)
        roi_x2 = paddle.index_select(roi, index=paddle.to_tensor(2), axis=1)
        roi_w = roi_x2 - roi_x1
        roi_y1 = paddle.index_select(roi, index=paddle.to_tensor(1), axis=1)
        roi_y2 = paddle.index_select(roi, index=paddle.to_tensor(3), axis=1)
        roi_h = roi_y2 - roi_y1
        roi_area = paddle.multiply(roi_w, roi_h)
        A = roi_area
        A1 = paddle.full(A.shape, 1, dtype='int32')
        A2 = paddle.where(A == 0, paddle.zeros_like(A1), A1)
        AE = paddle.expand(A2, [A.shape[0], x.shape[1]])
        rois_num = paddle.to_tensor(rois_num, dtype='int32')
        if self.dropout_before_pool and self.dropout_ratio > 0 :
            x = self.dropout(x)
        x = self.temporal_pool(x)
        x = self.spatial_pool(x)
        if not self.dropout_before_pool and self.dropout_ratio > 0 :
            x = self.dropout(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = paddle.multiply(x, paddle.cast(AE,"float32"))
        cls_score = self.fc_cls(x)
        # We do not predict bbox, so return None
        return cls_score, None

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, pos_weight):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        cls_reg_targets = self.bbox_target(pos_proposals, neg_proposals,
                                      pos_gt_labels, pos_weight)
        return cls_reg_targets

    def bbox_target(self, pos_bboxes_list, neg_bboxes_list, gt_labels, pos_weight):
        """Generate classification targets for bboxes.  """
        labels, label_weights = [], []
        pos_weight = 1.0 if pos_weight <= 0 else pos_weight
    
        assert len(pos_bboxes_list) == len(neg_bboxes_list) == len(gt_labels)
        length = len(pos_bboxes_list)
    
        for i in range(length):
            pos_bboxes = pos_bboxes_list[i]
            neg_bboxes = neg_bboxes_list[i]
            gt_label = gt_labels[i]
            num_pos = pos_bboxes.shape[0]
            if neg_bboxes is not None:
                num_neg = neg_bboxes.shape[0]
            else:
                num_neg = 0
            num_samples = num_pos + num_neg
            neg_label = paddle.zeros([num_neg, gt_label.shape[1]])
            label = paddle.concat([gt_label,neg_label])
            labels.append(label)
    
        labels = paddle.concat(labels, 0)
        return labels

    def recall_prec(self, pred_vec, target_vec):
        correct = paddle.to_tensor(np.logical_and(pred_vec.numpy(), target_vec.numpy()))
        correct = paddle.where(correct, 
                                    paddle.full(correct.shape,1,dtype='int32'),
                                    paddle.full(correct.shape,0,dtype='int32'))
        recall_correct = paddle.cast(paddle.sum(correct, axis=1), 'float32')
        target_vec = paddle.where(target_vec, 
                                    paddle.full(target_vec.shape,1,dtype='int32'),
                                    paddle.full(target_vec.shape,0,dtype='int32'))
        recall_target = paddle.cast(paddle.sum(target_vec, axis=1),'float32')
        recall = recall_correct / recall_target
        pred_vec = paddle.where(pred_vec, 
                                    paddle.full(pred_vec.shape,1,dtype='int32'),
                                    paddle.full(pred_vec.shape,0,dtype='int32'))
        prec_target = paddle.cast(paddle.sum(pred_vec, axis=1) + 1e-6, 'float32')
        prec = recall_correct / prec_target
        recall_mean = paddle.mean(recall)
        prec_mean = paddle.mean(prec)
        return recall_mean, prec_mean

    def multilabel_accuracy(self, pred, target, thr=0.5):
        pred = paddle.nn.functional.sigmoid(pred)
        pred_vec = pred > thr
        target_vec = target > 0.5
        recall_thr, prec_thr = self.recall_prec(pred_vec, target_vec)
        recalls, precs = [], []
        for k in self.topk:
            _, pred_label = paddle.topk(pred, k, 1, True, True)
            pred_vec = paddle.full(pred.shape,0,dtype='bool')
            num_sample = pred.shape[0]
            for i in range(num_sample):
                pred_vec[i, pred_label[i].numpy()] = 1  
            recall_k, prec_k = self.recall_prec(pred_vec, target_vec)
            recalls.append(recall_k)
            precs.append(prec_k)
        return recall_thr, prec_thr, recalls, precs

    def loss(self,
             cls_score,
             labels):
        losses = dict()
        if cls_score is not None:
            # Only use the cls_score
            labels = labels[:, 1:]
            pos_inds_bool = paddle.sum(labels, axis=-1) > 0
            pos_inds = paddle.where(paddle.sum(labels, axis=-1) > 0,
                                    paddle.full([labels.shape[0]],1,dtype='int32'),
                                    paddle.full([labels.shape[0]],0,dtype='int32'))
            pos_inds = paddle.nonzero(pos_inds, as_tuple=False)
            cls_score = paddle.index_select(cls_score, pos_inds, axis=0)
            cls_score = cls_score[:, 1:] 
            labels = paddle.index_select(labels, pos_inds, axis=0)
            bce_loss = F.binary_cross_entropy_with_logits
            loss = bce_loss(cls_score, labels, reduction='none')
            losses['loss'] = paddle.mean(loss)
            recall_thr, prec_thr, recall_k, prec_k = self.multilabel_accuracy(
                cls_score, labels, thr=0.5)
            losses['recall@thr=0.5'] = recall_thr
            losses['prec@thr=0.5'] = prec_thr
            for i, k in enumerate(self.topk):
                losses[f'recall@top{k}'] = recall_k[i]
                losses[f'prec@top{k}'] = prec_k[i]
        return losses

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        assert self.multilabel
        m = paddle.nn.Sigmoid()
        scores = m(cls_score)
        bboxes = rois
        return bboxes, scores
