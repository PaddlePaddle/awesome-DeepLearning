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
import copy
import os.path as osp

from ..registry import DATASETS
from .base import BaseDataset


@DATASETS.register()
class FeatureDataset(BaseDataset):
    """Feature dataset for action recognition
       Example:(TODO)
       Args:(TODO)
    """
    def __init__(
        self,
        file_path,
        pipeline,
        data_prefix=None,
        test_mode=False,
        suffix=None,
    ):
        self.suffix = suffix
        super().__init__(file_path, pipeline, data_prefix, test_mode)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        with open(self.file_path, 'r') as fin:
            for line in fin:
                filename = line.strip()
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                if self.suffix is not None:
                    filename = filename + self.suffix

                info.append(dict(filename=filename))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)

        return results['rgb_data'], results['rgb_len'], results[
            'rgb_mask'], results['audio_data'], results['audio_len'], results[
                'audio_mask'], results['labels']

    def prepare_test(self, idx):
        """TEST. Prepare the data for testing given the index."""
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)

        return results['rgb_data'], results['rgb_len'], results[
            'rgb_mask'], results['audio_data'], results['audio_len'], results[
                'audio_mask'], results['labels']
