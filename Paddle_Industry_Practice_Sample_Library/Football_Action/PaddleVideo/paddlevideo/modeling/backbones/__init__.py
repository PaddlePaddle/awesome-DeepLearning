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

from .actbert import BertForMultiModalPreTraining
from .adds import ADDS_DepthNet
from .agcn import AGCN
from .bmn import BMN
from .cfbi import CFBI
from .resnet import ResNet
from .resnet_slowfast import ResNetSlowFast
from .resnet_slowfast_MRI import ResNetSlowFast_MRI
from .resnet_tsm import ResNetTSM
from .resnet_tsm_MRI import ResNetTSM_MRI
from .resnet_tsn_MRI import ResNetTSN_MRI
from .resnet_tweaks_tsm import ResNetTweaksTSM
from .resnet_tweaks_tsn import ResNetTweaksTSN
from .stgcn import STGCN
from .swin_transformer import SwinTransformer3D
from .transnetv2 import TransNetV2
from .vit import VisionTransformer
from .vit_tweaks import VisionTransformer_tweaks

__all__ = [
    'ResNet', 'ResNetTSM', 'ResNetTweaksTSM', 'ResNetSlowFast', 'BMN',
    'ResNetTweaksTSN', 'VisionTransformer', 'STGCN', 'AGCN', 'TransNetV2',
    'SwinTransformer3D', 'BertForMultiModalPreTraining', 'ADDS_DepthNet',
    'VisionTransformer_tweaks', 'ResNetTSN_MRI', 'ResNetTSM_MRI',
    'ResNetSlowFast_MRI', 'CFBI'
]
