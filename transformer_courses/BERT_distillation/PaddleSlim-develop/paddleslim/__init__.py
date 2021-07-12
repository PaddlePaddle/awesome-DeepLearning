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
from paddleslim import models
from paddleslim import prune
from paddleslim import nas
from paddleslim import analysis
from paddleslim import dist
from paddleslim import quant
from paddleslim import pantheon
from paddleslim import dygraph
__all__ = [
    'models', 'prune', 'nas', 'analysis', 'dist', 'quant', 'pantheon', 'dygraph'
]

from paddleslim.dygraph import *
__all__ += dygraph.__all__
from paddleslim.analysis import *
__all__ += analysis.__all__
