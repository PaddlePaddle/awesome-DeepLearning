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
import json

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger
logger = get_logger("paddlevideo")


@DATASETS.register()
class BMNDataset(BaseDataset):
    """Video dataset for action localization.
    """
    def __init__(
        self,
        file_path,
        pipeline,
        subset,
        **kwargs,
    ):
        self.subset = subset
        super().__init__(file_path, pipeline, **kwargs)

    def load_file(self):
        """Load index file to get video information."""
        info = []
        annos = json.load(open(self.file_path))
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if self.subset in video_subset:
                info.append(
                    dict(
                        video_name=video_name,
                        video_info=annos[video_name],
                    ))
        #sort by video_name
        sort_f = lambda elem: elem['video_name']
        info.sort(key=sort_f)
        #add video_idx to info
        for idx, elem in enumerate(info):
            info[idx]['video_idx'] = idx
        logger.info("{} subset video numbers: {}".format(
            self.subset, len(info)))
        return info

    def prepare_train(self, idx):
        """TRAIN & VALID: Prepare data for training/valid given the index."""
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        return results['video_feat'], results['gt_iou_map'], results['gt_start'],\
               results['gt_end']

    def prepare_test(self, idx):
        """TEST: Prepare the data for test given the index."""
        results = copy.deepcopy(self.info[idx])
        results = self.pipeline(results)
        return results['video_feat'], results['gt_iou_map'], results['gt_start'], \
               results['gt_end'], results['video_idx']
