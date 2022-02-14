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
import numpy as np
from ..registry import ROI_EXTRACTORS
from .roi_extractor import RoIAlign


@ROI_EXTRACTORS.register()
class SingleRoIExtractor3D(nn.Layer):
    """Extract RoI features from a single level feature map.  """

    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 with_temporal_pool=True,
                 with_global=False):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.with_temporal_pool = with_temporal_pool
        self.with_global = with_global

        self.roi_layer = RoIAlign(resolution=self.output_size,
                                  spatial_scale=self.spatial_scale,
                                  sampling_ratio=self.sampling_ratio,
                                  aligned=self.aligned)

    def init_weights(self):
        pass

    # The shape of feat is N, C, T, H, W
    def forward(self, feat, rois, rois_num):
        if len(feat) >= 2:
            assert self.with_temporal_pool
        if self.with_temporal_pool:
            xi = 0
            for x in feat:
                xi = xi + 1
                y = paddle.mean(x, 2, keepdim=True)
            feat = [paddle.mean(x, 2, keepdim=True) for x in feat]
        feat = paddle.concat(feat, axis=1)  # merge slow and fast
        roi_feats = []
        for t in range(feat.shape[2]):
            if type(t) == paddle.fluid.framework.Variable:
                index = paddle.to_tensor(t)
            else:
                data_index = np.array([t]).astype('int32')
                index = paddle.to_tensor(data_index)

            frame_feat = paddle.index_select(feat, index, axis=2)
            frame_feat = paddle.squeeze(frame_feat,
                                        axis=2)  #axis=2,避免N=1时, 第一维度被删除.
            roi_feat = self.roi_layer(frame_feat, rois, rois_num)
            roi_feats.append(roi_feat)

        ret = paddle.stack(roi_feats, axis=2)
        return ret
