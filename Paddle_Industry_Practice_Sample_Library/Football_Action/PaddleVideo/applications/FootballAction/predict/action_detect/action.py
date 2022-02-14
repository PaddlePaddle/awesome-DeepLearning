#!./python27-gcc482/bin/python
# coding: utf-8
"""
BAIDU CLOUD action
"""

import os
import sys
import pickle
import json
import time
import functools

import numpy as np

from utils.preprocess import get_images
from utils.config_utils import parse_config, print_configs
import mfcc.feature_extractor as mfcc_extractor

import models.pptsm_infer as image_model
import models.audio_infer as audio_model
import models.bmn_infer as prop_model
import models.lstm_infer as classify_model

import logger
logger = logger.Logger()

def record_time_info(func):
    """decorator func to log cost time for func
    """
    @functools.wraps(func)
    def timer(*args):
        """log cost time for func
        """
        logger.info("function [{}] processing ...".format(func.__name__))
        start_time = time.time()
        retval = func(*args)
        cost_time = round(time.time() - start_time, 5)
        logger.info("function [{}] run time: {:.2f} min".format(func.__name__, cost_time / 60))
        return retval
    return timer


class ActionDetection(object):
    """ModelPredict"""
    def __init__(self, cfg_file="configs/configs.yaml"):
        cfg = parse_config(cfg_file)
        self.configs = cfg
        print_configs(self.configs, "Infer")

        name = 'COMMON'
        self.DEBUG          = cfg[name]['DEBUG']
        self.BMN_ONLY       = cfg[name]['BMN_ONLY']
        self.LSTM_ONLY      = cfg[name]['LSTM_ONLY']
        self.PCM_ONLY       = cfg[name]['PCM_ONLY']
        if self.LSTM_ONLY:
            self.prop_dict = {}
            for dataset in ['EuroCup2016']:
                prop_json = '/home/work/datasets/{}/feature_bmn/prop.json'.format(dataset)
                json_data = json.load(open(prop_json, 'r'))
                for item in json_data:
                    basename = prop_json.replace('feature_bmn/prop.json', 'mp4')
                    basename = basename + '/' + item['video_name'] + '.mp4'
                    self.prop_dict[basename] = item['bmn_results']
            

    @record_time_info
    def load_model(self):
        """
        load_model
        """
        if not self.DEBUG:
            self.image_model = image_model.InferModel(self.configs)
            if not self.PCM_ONLY:
                self.audio_model = audio_model.InferModel(self.configs)
    
        if not self.LSTM_ONLY:
            self.prop_model = prop_model.InferModel(self.configs)

        if not self.BMN_ONLY:
            self.classify_model = classify_model.InferModel(self.configs)

        logger.info("==> Action Detection prepared.")

    @record_time_info
    def infer(self, imgs_path, pcm_path, fps=5):
        """
        extract_feature
        """
        self.imgs_path = imgs_path
        self.pcm_path = pcm_path
        self.configs['COMMON']['fps'] = fps

        logger.info("==> input video {}".format(os.path.basename(self.imgs_path)))
    
        # step 1: extract feature
        video_features = self.extract_feature()
    
        # step2: get proposal
        bmn_results = self.extract_proposal(video_features)
         
        # step3: classify 
        material = {'feature': video_features, 'proposal': bmn_results}
        action_results = self.video_classify(material)
        
        return bmn_results, action_results

    @record_time_info
    def video_classify(self, material):
        """video classify"""
        if self.BMN_ONLY:
            return []
        action_results = self.classify_model.predict(self.configs, material=material) 
        logger.info('action shape {}'.format(np.array(action_results).shape))
        return action_results

    @record_time_info
    def extract_proposal(self, video_features):
        """extract proposal"""
        if self.LSTM_ONLY:
            basename = self.imgs_path.replace('frames', 'mp4') + '.mp4'
            bmn_results = self.prop_dict[basename]
            return bmn_results
        bmn_results = self.prop_model.predict(self.configs, material=video_features)
        logger.info('proposal shape {}'.format(np.array(bmn_results).shape))
        return bmn_results

    @record_time_info
    def extract_feature(self):
        """extract feature"""
        if not self.DEBUG:
            image_path_list = get_images(self.imgs_path)
            self.configs['PPTSM']['frame_list'] = image_path_list
            self.configs['AUDIO']['pcm_file'] = self.pcm_path
            image_features = self.image_model.predict(self.configs)
            if self.PCM_ONLY:
                sample_rate = self.configs['AUDIO']['sample_rate']
                pcm_features = mfcc_extractor.extract_pcm(self.pcm_path, sample_rate)
                audio_features = []
            else:
                audio_features, pcm_features = self.audio_model.predict(self.configs)

            np_image_features = np.array(image_features, dtype=np.float32)
            np_audio_features = np.array(audio_features, dtype=np.float32)
            np_pcm_features = np.array(pcm_features, dtype=np.float32)

            video_features = {'image_feature': np_image_features,
                              'audio_feature': np_audio_features,
                              'pcm_feature': np_pcm_features}
        else:
            feature_path = self.imgs_path.replace("frames", "features") + '.pkl'
            video_features = pickle.load(open(feature_path, 'rb'))

        logger.info("feature shape {} {} {}".format(video_features['image_feature'].shape,
                                                    video_features['audio_feature'].shape,
                                                    video_features['pcm_feature'].shape))

        return video_features

if __name__ == '__main__':

    model_predict = ActionDetection(cfg_file="../configs/configs.yaml")
    model_predict.load_model()

    imgs_path = "/home/work/datasets/EuroCup2016/frames/1be705a8f67648da8ec4b4296fa80895"
    pcm_path = "/home/work/datasets/EuroCup2016/pcm/1be705a8f67648da8ec4b4296fa80895.pcm"

    bmn_results, action_results = model_predict.infer(imgs_path, pcm_path)
    results = {'bmn_results': bmn_results, 'action_results': action_results}

    with open('results.json', 'w', encoding='utf-8') as f:
       data = json.dumps(results, indent=4, ensure_ascii=False)
       f.write(data)

