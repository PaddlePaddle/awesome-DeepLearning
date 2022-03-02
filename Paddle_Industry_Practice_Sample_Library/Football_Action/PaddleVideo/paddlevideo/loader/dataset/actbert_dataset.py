# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import lmdb
import pickle
import json

from paddlenlp.transformers import BertTokenizer
from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class ActBertDataset(BaseDataset):
    """ActBert dataset.
    """
    def __init__(
        self,
        file_path,
        pipeline,
        bert_model="bert-base-uncased",
        data_prefix=None,
        test_mode=False,
    ):
        self.bert_model = bert_model
        super().__init__(file_path, pipeline, data_prefix, test_mode)

    def load_file(self):
        """Load index file to get video information."""
        feature_data = np.load(self.file_path, allow_pickle=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model,
                                                       do_lower_case=True)
        self.info = []
        for item in feature_data:
            self.info.append(dict(feature=item, tokenizer=self.tokenizer))
        return self.info

    def prepare_train(self, idx):
        """Prepare the frames for training/valid given index. """
        results = copy.deepcopy(self.info[idx])
        #print('==results==', results)
        results = self.pipeline(results)
        return results['features']

    def prepare_test(self, idx):
        """Prepare the frames for test given index. """
        pass
