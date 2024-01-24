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
from ..registry import BBOX_SAMPLERS

class SamplingResult():
    """Bbox sampling result.  """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = paddle.index_select(bboxes,pos_inds)
        
        # neg_inds may be empty
        if neg_inds.shape[0]!=0:
            self.neg_bboxes = paddle.index_select(bboxes,neg_inds)
        else:
            self.neg_bboxes=None
        
        self.pos_is_gt  = paddle.index_select(gt_flags,pos_inds)
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = paddle.index_select(assign_result.gt_inds,pos_inds) - 1

        if gt_bboxes.numel().numpy()[0] == 0:
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = paddle.index_select(gt_bboxes, self.pos_assigned_gt_inds)

        if assign_result.labels is not None:
            self.pos_gt_labels = paddle.index_select(assign_result.labels, pos_inds)
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        if self.neg_bboxes is not None:
            ret = paddle.concat([self.pos_bboxes, self.neg_bboxes])
        else:
            # neg bbox may be empty
            ret = self.pos_bboxes
        return ret



@BBOX_SAMPLERS.register()
class RandomSampler():
    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
 
    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.  """

        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = paddle.full([bboxes.shape[0], ], 0, dtype='int32')
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = paddle.concat([gt_bboxes, bboxes])
            assign_result.add_gt_(gt_labels)
            gt_ones = paddle.full([gt_bboxes.shape[0], ], 1, dtype='int32')
            gt_flags = paddle.concat([gt_ones, gt_flags])

        #1. 得到正样本的数量, inds
        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self._sample_pos( assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        pos_inds = paddle.to_tensor(np.unique(pos_inds.numpy()))

        #2. 得到负样本的数量, inds
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        neg_inds = self._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = paddle.to_tensor(np.unique(neg_inds.numpy()))

        #3. 得到sampling result
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.  """
        assert len(gallery) >= num

        perm = paddle.arange(gallery.numel())[:num]
        perm = paddle.randperm(gallery.numel())[:num] 
        rand_inds = paddle.index_select(gallery, perm)
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        #1.首先看一下给的bboxes里面有哪些label是大于0的 得到了他们的index
        pos_inds = paddle.nonzero(assign_result.gt_inds, as_tuple=False)

        #2. 只要这个pos_inds的数目不是0个 这些就都可以是positive sample
        # 当pos_inds的数目小于num_expected(想要的sample的最大数目), 就直接用这个pos_inds
        # 反之就从这么多index里随机采样num_expected个出来
        if pos_inds.numel().numpy()[0] != 0:
            pos_inds = pos_inds.squeeze() 
        if pos_inds.numel().numpy()[0] <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = paddle.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel().numpy()[0] != 0:
            neg_inds = neg_inds.squeeze() 
        if (neg_inds.numel().numpy()[0]) <= num_expected.numpy()[0]:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
