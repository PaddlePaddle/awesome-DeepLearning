# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from .mobilenetv2 import MobileNetV2Space
from .mobilenetv1 import MobileNetV1Space
from .resnet import ResNetSpace
from .mobilenet_block import MobileNetV1BlockSpace, MobileNetV2BlockSpace
from .resnet_block import ResNetBlockSpace
from .inception_block import InceptionABlockSpace, InceptionCBlockSpace
from .darts_space import DartsSpace
from .search_space_registry import SEARCHSPACE
from .search_space_factory import SearchSpaceFactory
from .search_space_base import SearchSpaceBase
__all__ = [
    'MobileNetV1Space', 'MobileNetV2Space', 'ResNetSpace', 'DartsSpace',
    'MobileNetV1BlockSpace', 'MobileNetV2BlockSpace', 'ResNetBlockSpace',
    'InceptionABlockSpace', 'InceptionCBlockSpace', 'SearchSpaceBase',
    'SearchSpaceFactory', 'SEARCHSPACE'
]
