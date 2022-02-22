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

import logging

import paddle
import paddle.version as fluid_version
from ..common import get_logger

_logger = get_logger(__name__, level=logging.INFO)

try:
    paddle.utils.require_version('1.8.4')
    version_installed = [
        fluid_version.major, fluid_version.minor, fluid_version.patch,
        fluid_version.rc
    ]
    assert version_installed != [
        '2', '0', '0-alpha0', '0'
    ], "training-aware and post-training quant is not supported in 2.0 alpha version paddle"
    from .quanter import quant_aware, convert, quant_post_static, quant_post_dynamic
    from .quanter import quant_post, quant_post_only_weight
except Exception as e:
    _logger.warning(e)
    _logger.warning(
        "If you want to use training-aware and post-training quantization, "
        "please use Paddle >= 1.8.4 or develop version")

from .quant_embedding import quant_embedding
