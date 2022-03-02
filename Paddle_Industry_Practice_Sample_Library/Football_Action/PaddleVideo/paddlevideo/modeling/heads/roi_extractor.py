#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from . import ops


#@register
class RoIAlign(object):

    def __init__(self,
                 resolution=14,
                 spatial_scale=0.0625,
                 sampling_ratio=0,
                 aligned=False):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def __call__(self, feats, roi, rois_num):
        roi = paddle.concat(roi) if len(roi) > 1 else roi[0]
        rois_num = paddle.to_tensor(rois_num, dtype='int32')
        rois_num = paddle.cast(rois_num, dtype='int32')
        if len(feats) == 1:
            roi_feat = ops.roi_align(feats,
                                     roi,
                                     self.resolution,
                                     self.spatial_scale,
                                     sampling_ratio=self.sampling_ratio,
                                     rois_num=rois_num,
                                     aligned=self.aligned)
        else:
            rois_feat_list = []
            roi_feat = ops.roi_align(feats,
                                     roi,
                                     self.resolution,
                                     self.spatial_scale,
                                     sampling_ratio=self.sampling_ratio,
                                     rois_num=rois_num,
                                     aligned=self.aligned)

        return roi_feat
