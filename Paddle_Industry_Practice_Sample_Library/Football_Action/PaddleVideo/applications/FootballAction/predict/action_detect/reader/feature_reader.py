"""
attention-lstm feature reader
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import sys
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
import numpy as np
import random
import code

from .reader_utils import DataReader

class FeatureReader(DataReader):
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """

    def __init__(self, name, mode, cfg, material=None):
        self.name = name
        self.mode = mode
        self.batch_size = cfg[self.name.upper()]['batch_size']

        self.feature = material['feature']
        self.proposal = material['proposal']
        self.fps = 5

    def create_reader(self):
        """
        create_reader
        """
        image_feature_list = self.feature['image_feature']
        audio_feature_list = self.feature['audio_feature']
        pcm_feature_list = self.feature['pcm_feature']
        pcm_feature_list = pcm_feature_list.reshape((pcm_feature_list.shape[0] * 5, 640))

        fl = self.proposal

        if self.mode == 'train':
            random.shuffle(fl)

        def reader():
            """
            reader
            """
            batch_out = []
            for prop_info in fl:
                start_id = int(prop_info['start'])
                end_id = int(prop_info['end'])
                bmn_score = float(prop_info['score'])
                try:
                    image_feature = image_feature_list[start_id: end_id]
                    audio_feature = audio_feature_list[int(start_id / self.fps): int(end_id / self.fps)]
                    pcm_feature = pcm_feature_list[start_id: end_id]

                    # image_feature = np.concatenate((image_feature, pcm_feature), axis=1)
                    
                    batch_out.append((image_feature, audio_feature, 0, prop_info))
                    if len(batch_out) == self.batch_size:
                        yield batch_out
                        batch_out = []
                except Exception as e:
                    continue
        return reader

