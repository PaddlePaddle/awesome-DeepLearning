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
from __future__ import absolute_import
from ..nas import search_space
from .search_space import *
from ..nas import sa_nas
from .sa_nas import *
from .rl_nas import *
from ..nas import darts
from .darts import *
from .ofa import *
from .gp_nas import *

__all__ = []
__all__ += sa_nas.__all__
__all__ += search_space.__all__
__all__ += rl_nas.__all__
__all__ += darts.__all__
__all__ += gp_nas.__all__
