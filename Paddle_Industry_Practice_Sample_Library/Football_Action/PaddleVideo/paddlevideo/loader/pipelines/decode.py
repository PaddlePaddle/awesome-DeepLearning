#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np
import av
import cv2
import pickle
import decord as de
import math
import random
from ..registry import PIPELINES


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:  # here
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:  # ignore
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


@PIPELINES.register()
class VideoDecoder(object):
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend='cv2',
                 mode='train',
                 sampling_rate=32,
                 num_seg=8,
                 num_clips=1,
                 target_fps=30):

        self.backend = backend
        # params below only for TimeSformer
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.num_seg = num_seg
        self.num_clips = num_clips
        self.target_fps = target_fps

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'video'
        results['backend'] = self.backend

        if self.backend == 'cv2':
            cap = cv2.VideoCapture(file_path)
            videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sampledFrames = []
            for i in range(videolen):
                ret, frame = cap.read()
                # maybe first frame is empty
                if ret == False:
                    continue
                img = frame[:, :, ::-1]
                sampledFrames.append(img)
            results['frames'] = sampledFrames
            results['frames_len'] = len(sampledFrames)

        elif self.backend == 'decord':
            container = de.VideoReader(file_path)
            frames_len = len(container)
            results['frames'] = container
            results['frames_len'] = frames_len

        elif self.backend == 'pyav':  # for TimeSformer
            if self.mode in ["train", "valid"]:
                clip_idx = -1
            elif self.mode in ["test"]:
                clip_idx = 0
            else:
                raise NotImplementedError

            container = av.open(file_path)

            num_clips = 1  # always be 1

            # decode process
            fps = float(container.streams.video[0].average_rate)

            frames_length = container.streams.video[0].frames
            duration = container.streams.video[0].duration

            if duration is None:
                # If failed to fetch the decoding information, decode the entire video.
                decode_all_video = True
                video_start_pts, video_end_pts = 0, math.inf
            else:
                decode_all_video = False
                start_idx, end_idx = get_start_end_idx(
                    frames_length,
                    self.sampling_rate * self.num_seg / self.target_fps * fps,
                    clip_idx, num_clips)
                timebase = duration / frames_length
                video_start_pts = int(start_idx * timebase)
                video_end_pts = int(end_idx * timebase)

            frames = None
            # If video stream was found, fetch video frames from the video.
            if container.streams.video:
                margin = 1024
                seek_offset = max(video_start_pts - margin, 0)

                container.seek(seek_offset,
                               any_frame=False,
                               backward=True,
                               stream=container.streams.video[0])
                tmp_frames = {}
                buffer_count = 0
                max_pts = 0
                for frame in container.decode(**{"video": 0}):
                    max_pts = max(max_pts, frame.pts)
                    if frame.pts < video_start_pts:
                        continue
                    if frame.pts <= video_end_pts:
                        tmp_frames[frame.pts] = frame
                    else:
                        buffer_count += 1
                        tmp_frames[frame.pts] = frame
                        if buffer_count >= 0:
                            break
                video_frames = [tmp_frames[pts] for pts in sorted(tmp_frames)]

                container.close()

                frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
                clip_sz = self.sampling_rate * self.num_seg / self.target_fps * fps

                start_idx, end_idx = get_start_end_idx(
                    len(frames),  # frame_len
                    clip_sz,
                    clip_idx if decode_all_video else
                    0,  # If decode all video, -1 in train and valid, 0 in test;
                    # else, always 0 in train, valid and test, as we has selected clip size frames when decode.
                    1)
                results['frames'] = frames
                results['frames_len'] = len(frames)
                results['start_idx'] = start_idx
                results['end_idx'] = end_idx
        else:
            raise NotImplementedError
        return results


@PIPELINES.register()
class FrameDecoder(object):
    """just parse results
    """
    def __init__(self):
        pass

    def __call__(self, results):
        results['format'] = 'frame'
        return results


@PIPELINES.register()
class MRIDecoder(object):
    """just parse results
    """
    def __init__(self):
        pass

    def __call__(self, results):
        results['format'] = 'MRI'
        return results


@PIPELINES.register()
class FeatureDecoder(object):
    """
        Perform feature decode operations.e.g.youtube8m
    """
    def __init__(self, num_classes, max_len=512, has_label=True):
        self.max_len = max_len
        self.num_classes = num_classes
        self.has_label = has_label

    def __call__(self, results):
        """
        Perform feature decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        #1. load pkl
        #2. parse to rgb/audio/
        #3. padding

        filepath = results['filename']
        data = pickle.load(open(filepath, 'rb'), encoding='bytes')

        record = data
        nframes = record['nframes'] if 'nframes' in record else record[
            b'nframes']
        rgb = record['feature'].astype(
            float) if 'feature' in record else record[b'feature'].astype(float)
        audio = record['audio'].astype(
            float) if 'audio' in record else record[b'audio'].astype(float)
        if self.has_label:
            label = record['label'] if 'label' in record else record[b'label']
            one_hot_label = self.make_one_hot(label, self.num_classes)

        rgb = rgb[0:nframes, :]
        audio = audio[0:nframes, :]

        rgb = self.dequantize(rgb,
                              max_quantized_value=2.,
                              min_quantized_value=-2.)
        audio = self.dequantize(audio,
                                max_quantized_value=2,
                                min_quantized_value=-2)

        if self.has_label:
            results['labels'] = one_hot_label.astype("float32")

        feat_pad_list = []
        feat_len_list = []
        mask_list = []
        vitem = [rgb, audio]
        for vi in range(2):  #rgb and audio
            if vi == 0:
                prefix = "rgb_"
            else:
                prefix = "audio_"
            feat = vitem[vi]
            results[prefix + 'len'] = feat.shape[0]
            #feat pad step 1. padding
            feat_add = np.zeros((self.max_len - feat.shape[0], feat.shape[1]),
                                dtype=np.float32)
            feat_pad = np.concatenate((feat, feat_add), axis=0)
            results[prefix + 'data'] = feat_pad.astype("float32")
            #feat pad step 2. mask
            feat_mask_origin = np.ones(feat.shape, dtype=np.float32)
            feat_mask_add = feat_add
            feat_mask = np.concatenate((feat_mask_origin, feat_mask_add),
                                       axis=0)
            results[prefix + 'mask'] = feat_mask.astype("float32")

        return results

    def dequantize(self,
                   feat_vector,
                   max_quantized_value=2.,
                   min_quantized_value=-2.):
        """
        Dequantize the feature from the byte format to the float format
        """

        assert max_quantized_value > min_quantized_value
        quantized_range = max_quantized_value - min_quantized_value
        scalar = quantized_range / 255.0
        bias = (quantized_range / 512.0) + min_quantized_value

        return feat_vector * scalar + bias

    def make_one_hot(self, label, dim=3862):
        one_hot_label = np.zeros(dim)
        one_hot_label = one_hot_label.astype(float)
        for ind in label:
            one_hot_label[int(ind)] = 1
        return one_hot_label
