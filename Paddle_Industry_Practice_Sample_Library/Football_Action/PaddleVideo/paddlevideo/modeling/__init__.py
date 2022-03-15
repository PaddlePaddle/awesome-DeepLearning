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

from .assigners import MaxIoUAssignerAVA
from .backbones import ResNet
from .builder import (build_backbone, build_head, build_localizer, build_loss,
                      build_recognizer)
from .framework.detectors import BaseDetector, FastRCNN, TwoStageDetector
from .framework.recognizers import BaseRecognizer, Recognizer2D
from .heads import (AVARoIHead, BaseHead, BBoxHeadAVA, SingleRoIExtractor3D,
                    TSNHead)
from .losses import CrossEntropyLoss
from .registry import (BACKBONES, DETECTORS, HEADS, LOCALIZERS, LOSSES,
                       PARTITIONERS, RECOGNIZERS, ROI_EXTRACTORS)
from .samplers import RandomSampler
from .weight_init import kaiming_normal_, trunc_normal_, weight_init_

__all__ = [
    'BACKBONES', 'HEADS', 'RECOGNIZERS', 'LOCALIZERS', 'PARTITIONERS', 'LOSSES',
    'build_recognizer', 'build_localizer', 'build_head', 'build_backbone',
    'build_loss', 'ResNet', 'TSNHead', 'BaseHead', 'BaseRecognizer',
    'Recognizer2d', 'CrossEntropyLoss', 'ROI_EXTRACTORS',
    'SingleRoIExtractor3D', 'AVARoIHead', 'BBoxHeadAVA', 'MaxIoUAssignerAVA',
    'RandomSampler', 'DETECTORS', 'kaiming_normal_', 'trunc_normal_',
    'weight_init_'
]
