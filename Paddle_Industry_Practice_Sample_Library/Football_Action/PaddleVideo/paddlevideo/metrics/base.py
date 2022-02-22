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

from abc import abstractmethod
import numpy as np
import paddle
from paddlevideo.utils import get_dist_info

from .registry import METRIC


class BaseMetric(object):
    def __init__(self, data_size, batch_size, log_interval=1, **kwargs):
        self.data_size = data_size
        self.batch_size = batch_size
        _, self.world_size = get_dist_info()
        self.log_interval = log_interval

    @abstractmethod
    def update(self):
        raise NotImplemented

    @abstractmethod
    def accumulate(self):
        raise NotImplemented
