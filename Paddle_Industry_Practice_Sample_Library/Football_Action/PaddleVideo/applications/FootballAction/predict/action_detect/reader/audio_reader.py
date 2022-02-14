"""
audio reader
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
import os
import _pickle as cPickle
#from .reader_utils import DataReader
try:
    import cPickle as pickle
    from cStringIO import StringIO
except ImportError:
    import pickle
    from io import BytesIO
import numpy as np
import random
import code

from .reader_utils import DataReader
import mfcc.feature_extractor as feature_extractor

class AudioReader(DataReader):
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

        # set batch size and file list
        self.sample_rate = cfg[self.name.upper()]['sample_rate']
        self.batch_size = cfg[self.name.upper()]['batch_size']
        self.pcm_file = cfg[self.name.upper()]['pcm_file']
        self.material = material

    def create_reader(self):
        """create_reader"""
        with open(self.pcm_file, "rb") as f:
            pcm_data = f.read()
        audio_data = np.fromstring(pcm_data, dtype=np.int16)
        examples = feature_extractor.wav_to_example(audio_data, self.sample_rate)
        # print(examples.shape)

        def reader():
            """reader"""
            batch_out = []
            batch_out_pre = []
        
            for audio in examples:
                # batch_out.append([audio])
                batch_out.append(audio)
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) > 0:
                yield batch_out
            
        return reader
