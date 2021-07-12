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

import numpy as np
from ..core import GraphWrapper

__all__ = ["model_size"]


def model_size(program):
    """
    Get total value numbers of all parameters.

    Args:
        program(fluid.Program): The program used to calculate model size.

    Returns:
        int: The total count of all parameters. 
    """
    size = 0
    for block in program.blocks:
        for param in block.all_parameters():
            size += np.product(param.shape)
    return size
