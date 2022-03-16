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
import logging

python_ver = sys.version_info
logger = logging.getLogger('LSTM')


class FeatureReader:
    """
    Data reader for youtube-8M dataset, which was stored as features extracted by prior networks
    This is for the three models: lstm, attention cluster, nextvlad

    dataset cfg: num_classes
                 batch_size
                 list
                 NextVlad only: eigen_file
    """
    def __init__(self, name, mode, cfg, bs_denominator):
        self.name = name
        self.mode = mode
        self.num_classes = cfg.MODEL.num_classes

        # set batch size and file list
        self.batch_size = cfg[mode.upper()]['batch_size']
        self.droplast = cfg[mode.upper()]['droplast']
        self.filelist = cfg[mode.upper()]['filelist']
        self.eigen_file = cfg.MODEL.get('eigen_file', None)
        self.seg_num = cfg.MODEL.get('seg_num', None)
        self.num_gpus = bs_denominator

    def create_reader(self):
        """create_reader"""
        fin = open(self.filelist, 'r')
        lines = fin.readlines()
        fin.close()
        data = []
        for line in lines:
            items = line.strip().split()
            data.append(items[0])

        if self.mode == 'train':
            random.shuffle(data)

        def reader():
            """reader"""
            batch_out = []
            batch_out_pre = []
            yield_cnt = 0
            for i in range(len(data)):
                record = data[i]
                try:
                    pkl_filepath = record
                    pkl_data = pickle.load(open(pkl_filepath, 'rb'))
                    rgb_feature = pkl_data['image_feature'].astype(float)
                    audio_feature = pkl_data['audio_feature'].astype(float)
                    #print(pkl_filepath)
                    #print(rgb_feature, audio_feature)
                    video = pkl_filepath
                    if self.mode != 'infer':
                        label_id_info = pkl_data['label_info']
                        label_cls = [label_id_info['label']]
                        label_one = int(label_cls[0])
                        if len(label_cls) > 1:
                            label_index = random.randint(0, 1)
                            label_one = int(label_cls[label_index])
                        #one_hot_label = make_one_hot(label_cls, self.num_classes)
                        iou_norm = float(label_id_info['norm_iou'])
                        batch_out.append(
                            (rgb_feature, audio_feature, label_one, iou_norm))
                    else:
                        batch_out.append(
                            (rgb_feature, audio_feature, pkl_filepath))

                    if len(batch_out) == self.batch_size:
                        yield_cnt += 1
                        yield batch_out
                        batch_out_pre = batch_out[:]
                        batch_out = []
                except Exception as e:
                    logger.warn("warning: load data filed {}".format(record))
            # padding:
            # 1.0<batch_out<bs:
            #     If some data remained(not yielded), we padding they to one full batch to yield,
            # and yield k times to other gpus.
            # 2.len(batch_out) == 0 and yield_cnt % self.num_gpus > 0):
            #     If one (or some, yield_cnt % self.num_gpus) batch has been yielded but other gpus are
            # empty, data from batch_out_pre are yielded to other gpus. If gpu=2 and yield_cnt % self.num_gpus
            # =1, then k=0, batch_out_new will be yielded once.
            if self.droplast == False and ((len(batch_out) > 0 and len(batch_out) < self.batch_size) or ( \
                len(batch_out) == 0 and yield_cnt % self.num_gpus > 0)):
                batch_out_new = batch_out[:]
                if len(batch_out_pre) == 0:
                    batch_out_pre = batch_out[:]
                # return last batch k times
                for i in range(self.num_gpus - yield_cnt % self.num_gpus - 1):
                    yield batch_out_pre

                len_batch_out_pre = len(batch_out_pre)
                while len(batch_out_new) < self.batch_size:
                    index = random.randint(0, len_batch_out_pre - 1)
                    batch_out_new.append(batch_out_pre[index])
                yield batch_out_new

        return reader


def make_one_hot(label, dim=16):
    one_hot_label = np.zeros(dim)
    one_hot_label = one_hot_label.astype(float)
    for ind in label:
        one_hot_label[int(ind)] = 1
    return one_hot_label
