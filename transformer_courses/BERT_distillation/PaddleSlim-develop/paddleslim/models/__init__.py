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

from __future__ import absolute_import
from .util import image_classification
from .slimfacenet import SlimFaceNet_A_x0_60, SlimFaceNet_B_x0_75, SlimFaceNet_C_x0_75
from .slim_mobilenet import SlimMobileNet_v1, SlimMobileNet_v2, SlimMobileNet_v3, SlimMobileNet_v4, SlimMobileNet_v5
from ..models import mobilenet
from .mobilenet import *
from ..models import resnet
from .resnet import *

__all__ = ["image_classification"]
__all__ += mobilenet.__all__
__all__ += resnet.__all__
