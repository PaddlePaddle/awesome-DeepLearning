# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import logging
import numpy as np
from .log_helper import get_logger

__all__ = ['cached_reader']

_logger = get_logger(__name__, level=logging.INFO)


def cached_reader(reader, sampled_rate, cache_path, cached_id):
    """
    Sample partial data from reader and cache them into local file system.

    Args:
        reader: Iterative data source.
        sampled_rate(float): The sampled rate used to sample partial data for evaluation. None means using all data in eval_reader. default: None.
        cache_path(str): The path to cache the sampled data.
        cached_id(int): The id of dataset sampled. Evaluations with same cached_id use the same sampled dataset. default: 0.
    """
    np.random.seed(cached_id)
    cache_path = os.path.join(cache_path, str(cached_id))
    _logger.debug('read data from: {}'.format(cache_path))

    def s_reader():
        if os.path.isdir(cache_path):
            for file_name in open(os.path.join(cache_path, "list")):
                yield np.load(
                    os.path.join(cache_path, file_name.strip()),
                    allow_pickle=True)
        else:
            os.makedirs(cache_path)
            list_file = open(os.path.join(cache_path, "list"), 'w')
            batch = 0
            dtype = None
            for data in reader():
                if batch == 0 or (np.random.uniform() < sampled_rate):
                    np.save(
                        os.path.join(cache_path, 'batch' + str(batch)), data)
                    list_file.write('batch' + str(batch) + '.npy\n')
                    batch += 1
                    yield data

    return s_reader
