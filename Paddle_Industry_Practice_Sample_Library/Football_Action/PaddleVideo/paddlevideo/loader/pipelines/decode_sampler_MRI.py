# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random

import numpy as np
from PIL import Image
import SimpleITK as sitk
import cv2

from ..registry import PIPELINES


@PIPELINES.register()
class SFMRI_DecodeSampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 num_seg,
                 seg_len,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False):
        self.num_seg = num_seg
        self.seg_len = seg_len
        self.valid_mode = valid_mode
        self.select_left = select_left
        self.dense_sample = dense_sample
        self.linspace_sample = linspace_sample

    def _get(self, frames_idx_s, frames_idx_f, results):

        frame_dir = results['frame_dir']
        imgs_s = []
        imgs_f = []
        MRI = sitk.GetArrayFromImage(sitk.ReadImage(frame_dir))
        for idx in frames_idx_s:
            item = MRI[idx]
            item = cv2.resize(item, (224, 224))
            imgs_s.append(item)

        for idx in frames_idx_f:
            item = MRI[idx]
            item = cv2.resize(item, (224, 224))
            imgs_f.append(item)

        results['imgs'] = [imgs_s, imgs_f]
        return results

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        average_dur1 = int(frames_len / self.num_seg[0])
        average_dur2 = int(frames_len / self.num_seg[1])
        frames_idx_s = []
        frames_idx_f = []
        if self.linspace_sample:
            if 'start_idx' in results and 'end_idx' in results:
                offsets_s = np.linspace(results['start_idx'],
                                        results['end_idx'], self.num_seg[0])
                offsets_f = np.linspace(results['start_idx'],
                                        results['end_idx'], self.num_seg[1])
            else:
                offsets_s = np.linspace(0, frames_len - 1, self.num_seg[0])
                offsets_f = np.linspace(0, frames_len - 1, self.num_seg[1])
            offsets_s = np.clip(offsets_s, 0, frames_len - 1).astype(np.int64)
            offsets_f = np.clip(offsets_f, 0, frames_len - 1).astype(np.int64)

            frames_idx_s = list(offsets_s)
            frames_idx_f = list(offsets_f)

            return self._get(frames_idx_s, frames_idx_f, results)

        if not self.select_left:
            if self.dense_sample:  # For ppTSM
                if not self.valid_mode:  # train
                    sample_pos = max(1, 1 + frames_len - 64)
                    t_stride1 = 64 // self.num_seg[0]
                    t_stride2 = 64 // self.num_seg[1]
                    start_idx = 0 if sample_pos == 1 else np.random.randint(
                        0, sample_pos - 1)
                    offsets_s = [(idx * t_stride1 + start_idx) % frames_len + 1
                                 for idx in range(self.num_seg[0])]
                    offsets_f = [(idx * t_stride2 + start_idx) % frames_len + 1
                                 for idx in range(self.num_seg[1])]
                    frames_idx_s = offsets_s
                    frames_idx_f = offsets_f
                else:
                    sample_pos = max(1, 1 + frames_len - 64)
                    t_stride1 = 64 // self.num_seg[0]
                    t_stride2 = 64 // self.num_seg[1]
                    start_list = np.linspace(0,
                                             sample_pos - 1,
                                             num=10,
                                             dtype=int)
                    offsets_s = []
                    offsets_f = []
                    for start_idx in start_list.tolist():
                        offsets_s += [
                            (idx * t_stride1 + start_idx) % frames_len + 1
                            for idx in range(self.num_seg[0])
                        ]
                    for start_idx in start_list.tolist():
                        offsets_f += [
                            (idx * t_stride2 + start_idx) % frames_len + 1
                            for idx in range(self.num_seg[1])
                        ]
                    frames_idx_s = offsets_s
                    frames_idx_f = offsets_f
            else:
                for i in range(self.num_seg[0]):
                    idx = 0
                    if not self.valid_mode:
                        if average_dur1 >= self.seg_len:
                            idx = random.randint(0, average_dur1 - self.seg_len)
                            idx += i * average_dur1
                        elif average_dur1 >= 1:
                            idx += i * average_dur1
                        else:
                            idx = i
                    else:
                        if average_dur1 >= self.seg_len:
                            idx = (average_dur1 - 1) // 2
                            idx += i * average_dur1
                        elif average_dur1 >= 1:
                            idx += i * average_dur1
                        else:
                            idx = i
                    for jj in range(idx, idx + self.seg_len):
                        frames_idx_s.append(jj)

                for i in range(self.num_seg[1]):
                    idx = 0
                    if not self.valid_mode:
                        if average_dur2 >= self.seg_len:
                            idx = random.randint(0, average_dur2 - self.seg_len)
                            idx += i * average_dur2
                        elif average_dur2 >= 1:
                            idx += i * average_dur2
                        else:
                            idx = i
                    else:
                        if average_dur2 >= self.seg_len:
                            idx = (average_dur2 - 1) // 2
                            idx += i * average_dur2
                        elif average_dur2 >= 1:
                            idx += i * average_dur2
                        else:
                            idx = i
                    for jj in range(idx, idx + self.seg_len):
                        frames_idx_f.append(jj)

            return self._get(frames_idx_s, frames_idx_f, results)

        else:  # for TSM
            if not self.valid_mode:
                if average_dur2 > 0:
                    offsets_s = np.multiply(list(range(
                        self.num_seg[0])), average_dur1) + np.random.randint(
                            average_dur1, size=self.num_seg[0])

                    offsets_f = np.multiply(list(range(
                        self.num_seg[1])), average_dur2) + np.random.randint(
                            average_dur2, size=self.num_seg[1])
                elif frames_len > self.num_seg[1]:
                    offsets_s = np.sort(
                        np.random.randint(frames_len, size=self.num_seg[0]))
                    offsets_f = np.sort(
                        np.random.randint(frames_len, size=self.num_seg[1]))
                else:
                    offsets_s = np.zeros(shape=(self.num_seg[0], ))
                    offsets_f = np.zeros(shape=(self.num_seg[1], ))
            else:
                if frames_len > self.num_seg[1]:
                    average_dur_float_s = frames_len / self.num_seg[0]
                    offsets_s = np.array([
                        int(average_dur_float_s / 2.0 + average_dur_float_s * x)
                        for x in range(self.num_seg[0])
                    ])
                    average_dur_float_f = frames_len / self.num_seg[1]
                    offsets_f = np.array([
                        int(average_dur_float_f / 2.0 + average_dur_float_f * x)
                        for x in range(self.num_seg[1])
                    ])
                else:
                    offsets_s = np.zeros(shape=(self.num_seg[0], ))
                    offsets_f = np.zeros(shape=(self.num_seg[1], ))

            frames_idx_s = list(offsets_s)
            frames_idx_f = list(offsets_f)

            return self._get(frames_idx_s, frames_idx_f, results)
