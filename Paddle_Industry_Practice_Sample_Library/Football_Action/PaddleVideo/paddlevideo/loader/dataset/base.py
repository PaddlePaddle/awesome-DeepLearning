# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import numpy as np
from abc import ABC, abstractmethod

import paddle
from paddle.io import Dataset


class BaseDataset(Dataset, ABC):
    """Base class for datasets

    All datasets should subclass it.
    All subclass should overwrite:

    - Method: `load_file`, load info from index file.
    - Method: `prepare_train`, providing train data.
    - Method: `prepare_test`, providing test data.

    Args:
        file_path (str): index file path.
        pipeline (Sequence XXX)
        data_prefix (str): directory path of the data. Default: None.
        test_mode (bool): whether to build test dataset. Default: False.

    """
    def __init__(self, file_path, pipeline, data_prefix=None, test_mode=False):
        super().__init__()
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.test_mode = test_mode
        self.pipeline = pipeline
        self.info = self.load_file()

    @abstractmethod
    def load_file(self):
        """load the video information from the index file path."""
        pass

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        #unsqueeze label to list
        return results['imgs'], np.array([results['labels']])

    def prepare_test(self, idx):
        """TEST: Prepare the data for test given the index."""
        #Note: For now, paddle.io.DataLoader cannot support dict type retval, so convert to list here
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        #unsqueeze label to list
        return results['imgs'], np.array([results['labels']])

    def __len__(self):
        """get the size of the dataset."""
        return len(self.info)

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        if self.test_mode:
            return self.prepare_test(idx)
        else:
            return self.prepare_train(idx)
