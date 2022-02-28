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

from .actbert_dataset import ActBertDataset
from .ava_dataset import AVADataset
from .bmn_dataset import BMNDataset
from .davis_dataset import DavisDataset
from .feature import FeatureDataset
from .frame import FrameDataset, FrameDataset_Sport
from .MRI import MRIDataset
from .MRI_SlowFast import SFMRIDataset
from .msrvtt import MSRVTTDataset
from .oxford import MonoDataset
from .skeleton import SkeletonDataset
from .slowfast_video import SFVideoDataset
from .video import VideoDataset

__all__ = [
    'VideoDataset', 'FrameDataset', 'SFVideoDataset', 'BMNDataset',
    'FeatureDataset', 'SkeletonDataset', 'AVADataset', 'MonoDataset',
    'MSRVTTDataset', 'ActBertDataset', 'DavisDataset', 'MRIDataset',
    'SFMRIDataset', 'FrameDataset_Sport'
]
