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

from .two_stage import TwoStageDetector
from ...registry import DETECTORS

@DETECTORS.register()
class FastRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 pretrained=None):
        super(FastRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
