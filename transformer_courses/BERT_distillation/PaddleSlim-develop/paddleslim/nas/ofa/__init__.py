#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .ofa import OFA, RunConfig, DistillConfig
from .convert_super import supernet
from .utils.special_config import *
from .get_sub_model import *

from .utils.utils import get_paddle_version
pd_ver = get_paddle_version()
if pd_ver == 185:
    from .layers_old import *
else:
    from .layers import *
