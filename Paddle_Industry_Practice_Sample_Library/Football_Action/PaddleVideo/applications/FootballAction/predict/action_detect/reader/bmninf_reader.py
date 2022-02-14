"""
# @File  : bmninf_reader.py  
# @Author: macaihong
# @Date  : 2019/12/15
# @Desc  :
"""

import os
import random
import pickle
import json
import numpy as np
import multiprocessing

import numpy as np

from .reader_utils import DataReader


def get_sw_prop(duration, window=200, step=10):
    """
    get_sw_prop
    """
    pr = []
    local_boxes = []
    for k in np.arange(0, duration - window + step, step):
        start_id = k
        end_id = min(duration, k + window)
        if end_id - start_id < window:
            start_id = end_id - window
        local_boxes = (start_id, end_id)
        pr.append(local_boxes)

    def valid_proposal(duration, span):
        """
        valid_proposal
        """
        # fileter proposals
        # a valid proposal should have at least one second in the video
        real_span = min(duration, span[1]) - span[0]
        return real_span >= 1

    pr = list(filter(lambda x: valid_proposal(duration, x), pr))
    return pr


class BMNINFReader(DataReader):
    """
    Data reader for BMN model, which was stored as features extracted by prior networks
    dataset cfg: feat_path, feature path,
                 tscale, temporal length of BM map,
                 dscale, duration scale of BM map,
                 anchor_xmin, anchor_xmax, the range of each point in the feature sequence,
                 batch_size, batch size of input data,
                 num_threads, number of threads of data processing
    """

    def __init__(self, name, mode, cfg, material=None):
        self.name = name
        self.mode = mode
        self.tscale = cfg[self.name.upper()]['tscale']  # 200
        self.dscale = cfg[self.name.upper()]['dscale']  # 200
        # self.subset = cfg[self.name.upper()]['subset']
        self.tgap = 1. / self.tscale
        self.step = cfg[self.name.upper()]['window_step']

        self.material = material
        src_feature = self.material

        image_feature = src_feature['image_feature']
        pcm_feature = src_feature['pcm_feature']
        pcm_feature = pcm_feature.reshape((pcm_feature.shape[0] * 5, 640))
        # print(rgb_feature.shape, audio_feature.shape, pcm_feature.shape)
        min_length = min(image_feature.shape[0], pcm_feature.shape[0])
        #if min_length == 0:
        #    continue
        image_feature = image_feature[:min_length, :]
        pcm_feature = pcm_feature[:min_length, :]
        self.features = np.concatenate((image_feature, pcm_feature), axis=1)

        self.duration = len(self.features)
        self.window = self.tscale

        self.get_dataset_dict()
        self.get_match_map()

        self.batch_size = cfg[self.name.upper()]['batch_size']
        if (mode == 'test') or (mode == 'infer'):
            self.num_threads = 1  # set num_threads as 1 for test and infer

    def get_dataset_dict(self):
        """
        get_dataset_dict
        """
        self.video_list = get_sw_prop(self.duration, self.window, self.step)

    def get_match_map(self):
        """
        get_match_map
        """
        match_map = []
        for idx in range(self.tscale):
            tmp_match_window = []
            xmin = self.tgap * idx
            for jdx in range(1, self.tscale + 1):
                xmax = xmin + self.tgap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)
        match_map = np.transpose(match_map, [1, 0, 2])
        match_map = np.reshape(match_map, [-1, 2])
        self.match_map = match_map
        self.anchor_xmin = [self.tgap * i for i in range(self.tscale)]
        self.anchor_xmax = [self.tgap * i for i in range(1, self.tscale + 1)]

    
    def load_file(self, video_wind):
        """
        load_file
        """
        start_feat_id = video_wind[0]
        end_feat_id = video_wind[1]
        video_feat = self.features[video_wind[0]: video_wind[1]]
        video_feat = video_feat.T
        video_feat = video_feat.astype("float32")
        return video_feat

    def create_reader(self):
        """
        reader creator for ctcn model
        """
        return self.make_infer_reader()

    def make_infer_reader(self):
        """
        reader for inference
        """
        def reader():
            """
            reader
            """
            batch_out = []
            # for video_name in self.video_list:
            for video_wind in self.video_list:
                video_idx = self.video_list.index(video_wind)
                video_feat = self.load_file(video_wind)
                batch_out.append((video_feat, video_wind, [self.duration, self.dscale]))

                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) > 0:
                yield batch_out

        return reader
