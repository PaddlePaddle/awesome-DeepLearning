#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import random
from ..registry import PIPELINES
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """
    def __init__(self, window_size, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        T = self.get_frame_num(data)
        if T == self.window_size:
            data_pad = data[:, :self.window_size, :, :]
        elif T < self.window_size:
            begin = random.randint(0, self.window_size -
                                   T) if self.random_pad else 0
            data_pad = np.zeros((C, self.window_size, V, M))
            data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        else:
            if self.random_pad:
                index = np.random.choice(T, self.window_size,
                                         replace=False).astype('int64')
            else:
                index = np.linspace(0, T, self.window_size).astype("int64")
            data_pad = data[:, index, :, :]

        results['data'] = data_pad
        return results


@PIPELINES.register()
class SkeletonNorm(object):
    """
    Normalize skeleton feature.
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data = data - data[:, :, 8:9, :]
        data = data[:self.axis, :, :, :]  # get (x,y) from (x,y, acc)
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1

        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class Iden(object):
    """
    Wrapper Pipeline
    """
    def __init__(self, label_expand=True):
        self.label_expand = label_expand

    def __call__(self, results):
        data = results['data']
        results['data'] = data.astype('float32')

        if 'label' in results and self.label_expand:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results
