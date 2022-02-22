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

import math
import numpy as np


def compute_downsample_num(input_size, output_size):
    downsample_num = 0
    while input_size > output_size:
        input_size = math.ceil(float(input_size) / 2.0)
        downsample_num += 1

    if input_size != output_size:
        raise NotImplementedError(
            'output_size must can downsample by input_size!!!')

    return downsample_num


def check_points(count, points):
    if points is None:
        return False
    else:
        if isinstance(points, list):
            return (True if count in points else False)
        else:
            return (True if count == points else False)


def get_random_tokens(range_table):
    tokens = []
    for idx, max_value in enumerate(range_table):
        tokens_idx = int(np.floor(range_table[idx] * np.random.rand(1)))
        tokens.append(tokens_idx)
    return tokens
