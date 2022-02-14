# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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
import random
from PIL import Image
from ..registry import PIPELINES
import os
import numpy as np
import io
import os.path as osp
from abc import ABCMeta, abstractmethod
import cv2
from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED
import inspect

imread_backend = 'cv2'
imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


@PIPELINES.register()
class SampleFrames:
    """Sample frames from the video. """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode. """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode. """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode. """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)
        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading. """
        total_frames = results['total_frames']
        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)
        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets
        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')
        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends. """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass

class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'r') as f:
            value_buf = f.read()
        return value_buf

class FileClient:
    """A general file client to access files in different backend. """

    _backends = {
        'disk': HardDiskBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        if backend not in self._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    @classmethod
    def _register_backend(cls, name, backend, force=False):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        cls._backends[name] = backend

    @classmethod
    def register_backend(cls, name, backend=None, force=False):
        """Register a backend to FileClient. """

        if backend is not None:
            cls._register_backend(name, backend, force=force)
            return

        def _register(backend_cls):
            cls._register_backend(name, backend_cls, force=force)
            return backend_cls

        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath):
        return self.client.get_text(filepath)

@PIPELINES.register()
class RawFrameDecode:
    """Load and decode frames with given indices. """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def _pillow2array(self,img, flag='color', channel_order='bgr'):
        """Convert a pillow image to numpy array. """

        channel_order = channel_order.lower()
        if channel_order not in ['rgb', 'bgr']:
            raise ValueError('channel order must be either "rgb" or "bgr"')

        if flag == 'unchanged':
            array = np.array(img)
            if array.ndim >= 3 and array.shape[2] >= 3:  # color image
                array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
        else:
            # If the image mode is not 'RGB', convert it to 'RGB' first.
            if img.mode != 'RGB':
                if img.mode != 'LA':
                    # Most formats except 'LA' can be directly converted to RGB
                    img = img.convert('RGB')
                else:
                    # When the mode is 'LA', the default conversion will fill in
                    #  the canvas with black, which sometimes shadows black objects
                    #  in the foreground.
                    #
                    # Therefore, a random color (124, 117, 104) is used for canvas
                    img_rgba = img.convert('RGBA')
                    img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                    img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
            if flag == 'color':
                array = np.array(img)
                if channel_order != 'rgb':
                    array = array[:, :, ::-1]  # RGB to BGR
            elif flag == 'grayscale':
                img = img.convert('L')
                array = np.array(img)
            else:
                raise ValueError(
                    'flag must be "color", "grayscale" or "unchanged", '
                    f'but got {flag}')
        return array

    def _imfrombytes(self,content, flag='color', channel_order='bgr'):#, backend=None):
        """Read an image from bytes. """

        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if isinstance(flag, str) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        suffix = results['suffix']
        #modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        for frame_idx in results['frame_inds']:
            frame_idx += offset
            filepath = osp.join(directory, suffix.format(frame_idx))
            img_bytes = self.file_client.get(filepath) #以二进制方式读取图片
            # Get frame with channel order RGB directly.

            cur_frame = self._imfrombytes(img_bytes, channel_order='rgb')
            imgs.append(cur_frame)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        # we resize the gt_bboxes and proposals to their real scale
        h, w = results['img_shape']
        scale_factor = np.array([w, h, w, h])
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            gt_bboxes_new = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes_new
        if 'proposals' in results and results['proposals'] is not None:
            proposals = results['proposals']
            proposals = (proposals * scale_factor).astype(np.float32)
            results['proposals'] = proposals
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str

@PIPELINES.register()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)

        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        #delta=(timestamp - timestamp_start) 为该帧距离15min视频开头有几秒
        #center_index=fps*delta为该帧距离15min视频开头有几帧
        #center_index+1是为了避免后续采样时出现负数? 
        #后续需要以center_index为中心前后采样视频帧片段
        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        results['frame_inds'] = np.array(frame_inds, dtype=np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str

