"""
audio model config
"""
import numpy as np

import mfcc.feature_extractor as feature_extractor


class ModelAudio(object):
    """
    modelAudio
    """
    def __init__(self, configs, use_gpu=1):
        self.use_gpu = use_gpu

        self.audio_fps = configs.COMMON.fps
        self.audio_feat_scale = configs.TSN.audio_scale
        self.sample_rate = 16000

    def predict_slice(self, wav_data, sample_rate):
        """
        audio predict
        """
        examples_batch = feature_extractor.wav_to_example(
            wav_data, sample_rate)[0]
        return examples_batch

    def predict_audio(self, audio_file):
        """
        predict_audio
        """
        audio_feature_list = []
        # read pcm
        sample_rate = self.sample_rate
        try:
            with open(audio_file, "rb") as f:
                pcm_data = f.read()
            audio_data = np.fromstring(pcm_data, dtype=np.int16)
            audio_status = "audio load success"
        except Exception as e:
            audio_data = []
            audio_status = "audio load failed"
        step = 1
        len_video = int(len(audio_data) / sample_rate)
        print(len_video)
        for i in range(0, len_video, step):
            audio_data_part = audio_data[i * sample_rate:(i + step) *
                                         sample_rate]
            feature_audio = self.predict_slice(audio_data_part, sample_rate)
            audio_feature_list.append(feature_audio)
        return audio_feature_list
