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

import random
import numpy as np
from PIL import Image
import decord as de
from ..registry import PIPELINES


@PIPELINES.register()
class DecodeSampler(object):
    """
    We use 'decord' for decode and sampling, which is faster than opencv.
    This is used in slowfast model.
    Args:
        num_frames(int): the number of frames we want to sample.
        sampling_rate(int): sampling rate for video data.
        target_fps(int): desired fps, default 30
        test_mode(bool): whether test or train/valid. In slowfast, we use multicrop when test.
    """
    def __init__(self,
                 num_frames,
                 sampling_rate,
                 default_sampling_rate=2,
                 target_fps=30,
                 test_mode=False):
        self.num_frames = num_frames
        self.orig_sampling_rate = self.sampling_rate = sampling_rate
        self.default_sampling_rate = default_sampling_rate
        self.target_fps = target_fps
        self.test_mode = test_mode

    def get_start_end_idx(self, video_size, clip_size, clip_idx,
                          temporal_num_clips):
        delta = max(video_size - clip_size, 0)
        if not self.test_mode:
            # Random temporal sampling.
            start_idx = random.uniform(0, delta)
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / temporal_num_clips
        end_idx = start_idx + clip_size - 1
        return start_idx, end_idx

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx:
            self.sampling_rate = random.randint(self.default_sampling_rate,
                                                self.orig_sampling_rate)

        filepath = results['filename']
        temporal_sample_index = results['temporal_sample_index']
        temporal_num_clips = results['temporal_num_clips']

        vr = de.VideoReader(filepath)
        videolen = len(vr)

        fps = vr.get_avg_fps()
        clip_size = self.num_frames * self.sampling_rate * fps / self.target_fps

        start_idx, end_idx = self.get_start_end_idx(videolen, clip_size,
                                                    temporal_sample_index,
                                                    temporal_num_clips)
        index = np.linspace(start_idx, end_idx, self.num_frames).astype("int64")
        index = np.clip(index, 0, videolen)

        frames_select = vr.get_batch(index)  #1 for buffer

        # dearray_to_img
        np_frames = frames_select.asnumpy()
        frames_select_list = []
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i]
            frames_select_list.append(Image.fromarray(imgbuf, mode='RGB'))
        results['imgs'] = frames_select_list
        return results
