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
import numpy as np
from ..registry import BBOX_ASSIGNERS
from ..bbox_utils import bbox_overlaps

class AssignResult():
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        """Add ground truth as assigned results.  """
        self_inds = paddle.arange(1, len(gt_labels) + 1, dtype="int32")
        gt_inds_squeeze = paddle.squeeze(self.gt_inds, axis=0)
        self.gt_inds = paddle.concat([self_inds, gt_inds_squeeze])
        gt_label_ones = paddle.full((len(gt_labels), ), 1, dtype='float32')
        max_overlaps_squeeze = paddle.squeeze(self.max_overlaps, axis=0)
        self.max_overlaps = paddle.concat([gt_label_ones, max_overlaps_squeeze])
        if self.labels is not None:
            self.labels = paddle.concat([gt_labels, self.labels])

@BBOX_ASSIGNERS.register()
class MaxIoUAssignerAVA():
    """Assign a corresponding gt bbox or background to each bbox.  """
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality

    def assign(self, 
               bboxes, 
               gt_bboxes, 
               gt_labels=None):
        """Assign gt to bboxes.  """
        overlaps = bbox_overlaps(gt_bboxes, bboxes)
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.  """
        num_gts, num_bboxes = overlaps.shape[0], overlaps.shape[1]
        # 1. assign -1
        assigned_gt_inds = paddle.full((num_bboxes, ), -1, dtype='int32')

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = paddle.topk(overlaps, k=1, axis=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = paddle.topk(overlaps, k=1, axis=1) 

        # 2. assign negative: below the negative inds are set to be 0
        match_labels = paddle.full(argmax_overlaps.shape, -1, dtype='int32')
        match_labels = paddle.where(max_overlaps < self.neg_iou_thr,
                            paddle.zeros_like(match_labels), match_labels)

        # 3. assign positive: above positive IoU threshold
        argmax_overlaps_int32 = paddle.cast(argmax_overlaps, 'int32')
        match_labels = paddle.where(max_overlaps >= self.pos_iou_thr,
                                argmax_overlaps_int32 + 1, match_labels)
        assigned_gt_inds = match_labels
        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds
            # assigned in Step 3. Thus, the assigned gt might not be the
            # best one for prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox
            # 1 & 2, bbox 1 will be assigned as the best target for bbox A
            # in step 3. However, if GT bbox 2's gt_argmax_overlaps = A,
            # bbox A's assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps.numpy()[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        equal_x_np = overlaps[i, :].numpy()
                        equal_y_np = gt_max_overlaps[i].numpy()
                        max_iou_inds = np.equal(equal_x_np, equal_y_np)
                        max_iou_inds = paddle.to_tensor(max_iou_inds)
                        max_iou_inds = paddle.reshape( max_iou_inds, [1,max_iou_inds.shape[0]] )
                        match_labels_gts = paddle.full(max_iou_inds.shape, i+1, dtype='int32')
                        match_labels = paddle.where(max_iou_inds, match_labels_gts, match_labels)
                        assigned_gt_inds = match_labels
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            # consider multi-class case (AVA)
            assert len(gt_labels[0]) > 1
            assigned_labels = paddle.full([num_bboxes, len(gt_labels[0])], 0, dtype='float32')
            assigned_gt_inds_reshape = assigned_gt_inds.reshape([assigned_gt_inds.shape[1]])
            pos_inds = paddle.nonzero( assigned_gt_inds_reshape , as_tuple=False)
            pos_inds_num = paddle.numel(pos_inds).numpy()[0]
            if pos_inds_num > 0:
                pos_inds = paddle.squeeze(pos_inds, axis = 1 )
                assigned_gt_inds_squeeze = paddle.squeeze(assigned_gt_inds, axis=0)
                assigned_gt_inds_select = paddle.index_select(assigned_gt_inds_squeeze, pos_inds) - 1
                gt_labels_select = paddle.index_select(gt_labels, assigned_gt_inds_select)
                A = assigned_gt_inds_squeeze
                X = assigned_gt_inds_squeeze - 1
                Y = paddle.zeros_like(X)
                if A.shape[0]==1:
                    if A.numpy()[0]>0:
                        T=X
                    else:
                        T=Y
                else:
                    T = paddle.where(A>0, X, Y)
                S = paddle.index_select(gt_labels, T)
                AE = paddle.expand(A, [S.shape[1], A.shape[0]]) 
                AET = paddle.transpose(AE, perm=[1, 0])
                R = paddle.where(AET>0, S, assigned_labels) 
                assigned_labels = R
        else:
            assigned_labels = None
        ret = AssignResult(
            num_gts,
            assigned_gt_inds,
            max_overlaps,
            labels=assigned_labels)
        return ret
