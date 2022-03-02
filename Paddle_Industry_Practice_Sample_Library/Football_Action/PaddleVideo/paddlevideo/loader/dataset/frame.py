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
class FrameDataset(BaseDataset):
    """Rawframe dataset for action recognition.
    The dataset loads raw frames from frame files, and apply specified transform operatation them.
    The indecx file is a text file with multiple lines, and each line indicates the directory of frames of a video, toatl frames of the video, and its label, which split with a whitespace.
    Example of an index file:

    .. code-block:: txt

        file_path-1 150 1
        file_path-2 160 1
        file_path-3 170 2
        file_path-4 180 2

    Args:
        file_path (str): Path to the index file.
        pipeline(XXX):
        data_prefix (str): directory path of the data. Default: None.
        test_mode (bool): Whether to bulid the test dataset. Default: False.
        suffix (str): suffix of file. Default: 'img_{:05}.jpg'.

    """
    def __init__(self,
                 file_path,
                 pipeline,
                 num_retries=5,
                 data_prefix=None,
                 test_mode=False,
                 suffix='img_{:05}.jpg'):
        self.num_retries = num_retries
        self.suffix = suffix
        super().__init__(file_path, pipeline, data_prefix, test_mode)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                frame_dir, frames_len, labels = line_split
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                info.append(
                    dict(frame_dir=frame_dir,
                         suffix=self.suffix,
                         frames_len=frames_len,
                         labels=int(labels)))
        return info

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        #Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['frame_dir'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], np.array([results['labels']])

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        #Try to catch Exception caused by reading missing frames files
        for ir in range(self.num_retries):
            try:
                results = copy.deepcopy(self.info[idx])
                results = self.pipeline(results)
            except Exception as e:
                #logger.info(e)
                if ir < self.num_retries - 1:
                    logger.info(
                        "Error when loading {}, have {} trys, will try again".
                        format(results['frame_dir'], ir))
                idx = random.randint(0, len(self.info) - 1)
                continue
            return results['imgs'], np.array([results['labels']])


@DATASETS.register()
class FrameDataset_Sport(BaseDataset):
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
           **kwargs: Keyword arguments for ```BaseDataset```.
    """
    def __init__(self, file_path, pipeline, num_retries=5, suffix='', **kwargs):
        self.num_retries = num_retries
        self.suffix = suffix
        super().__init__(file_path, pipeline, **kwargs)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                frame_dir = line_split[0]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                info.append(dict(frame_dir=frame_dir, suffix=self.suffix))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
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
            return results['imgs'], np.array([results['labels']])

    def prepare_test(self, idx):
        """TEST. Prepare the data for test given the index."""
        #Try to catch Exception caused by reading corrupted video file
        for ir in range(self.num_retries):
            try:
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
            return results['imgs'], np.array([results['labels']])
