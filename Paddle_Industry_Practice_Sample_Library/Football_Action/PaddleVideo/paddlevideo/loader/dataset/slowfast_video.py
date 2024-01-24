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
import random
import numpy as np

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")

@DATASETS.register()
class SFVideoDataset(BaseDataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.

       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:

       .. code-block:: txt

           path/000.mp4 1
           path/001.mp4 1
           path/002.mp4 2
           path/003.mp4 2

       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           num_ensemble_views(int): temporal segment when multi-crop test
           num_spatial_crops(int): spatial crop number when multi-crop test
           **kwargs: Keyword arguments for ```BaseDataset```.

    """
    def __init__(
        self,
        file_path,
        pipeline,
        num_ensemble_views=1,
        num_spatial_crops=1,
        num_retries=5,
        num_samples_precise_bn=None,
        **kwargs,
    ):
        self.num_ensemble_views = num_ensemble_views
        self.num_spatial_crops = num_spatial_crops
        self.num_retries = num_retries
        self.num_samples_precise_bn = num_samples_precise_bn
        super().__init__(file_path, pipeline, **kwargs)
        #set random seed
        random.seed(0)
        np.random.seed(0)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                filename, labels = line_split
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                for tidx in range(self.num_ensemble_views):
                    for sidx in range(self.num_spatial_crops):
                        info.append(
                            dict(
                                filename=filename,
                                labels=int(labels),
                                temporal_sample_index=tidx,
                                spatial_sample_index=sidx,
                                temporal_num_clips=self.num_ensemble_views,
                                spatial_num_clips=self.num_spatial_crops,
                            ))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training given the index."""
        #Try to catch Exception caused by reading corrupted video file
        short_cycle = False
        if isinstance(idx, tuple):
            idx, short_cycle_idx = idx
            short_cycle = True
        for ir in range(self.num_retries):
            try:
                #Multi-grid short cycle
                if short_cycle:
                    results = copy.deepcopy(self.info[idx])
                    results['short_cycle_idx'] = short_cycle_idx
                else:
                    results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue

            return results['imgs'][0], results['imgs'][1], np.array(
                [results['labels']])

    def prepare_test(self, idx):
        """TEST. Prepare the data for test given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['filename'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'][0], results['imgs'][1], np.array(
                [results['labels']]), np.array([idx])

    def __len__(self):
        """get the size of the dataset."""
        if self.num_samples_precise_bn is None:
            return len(self.info)
        else:
            random.shuffle(self.info)
            return min(self.num_samples_precise_bn, len(self.info))
