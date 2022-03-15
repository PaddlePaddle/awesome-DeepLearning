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

import math
import random
from collections.abc import Sequence

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from PIL import Image

from ..registry import PIPELINES


@PIPELINES.register()
class Scale(object):
    """
    Scale images.
    Args:
        short_size(float | int): Short size of an image will be scaled to the short_size.
        fixed_ratio(bool): Set whether to zoom according to a fixed ratio. default: True
        do_round(bool): Whether to round up when calculating the zoom ratio. default: False
        backend(str): Choose pillow or cv2 as the graphics processing backend. default: 'pillow'
    """
    def __init__(self,
                 short_size,
                 fixed_ratio=True,
                 keep_ratio=None,
                 do_round=False,
                 backend='pillow'):
        self.short_size = short_size
        assert (fixed_ratio and not keep_ratio) or (not fixed_ratio), \
            f"fixed_ratio and keep_ratio cannot be true at the same time"
        self.fixed_ratio = fixed_ratio
        self.keep_ratio = keep_ratio
        self.do_round = do_round

        assert backend in [
            'pillow', 'cv2'
        ], f"Scale's backend must be pillow or cv2, but get {backend}"
        self.backend = backend

    def __call__(self, results):
        """
        Performs resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        imgs = results['imgs']
        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            if isinstance(img, np.ndarray):
                h, w, _ = img.shape
            elif isinstance(img, Image.Image):
                w, h = img.size
            else:
                raise NotImplementedError

            if w <= h:
                ow = self.short_size
                if self.fixed_ratio:
                    oh = int(self.short_size * 4.0 / 3.0)
                elif not self.keep_ratio:  # no
                    oh = self.short_size
                else:
                    scale_factor = self.short_size / w
                    oh = int(h * float(scale_factor) +
                             0.5) if self.do_round else int(h *
                                                            self.short_size / w)
                    ow = int(w * float(scale_factor) +
                             0.5) if self.do_round else int(w *
                                                            self.short_size / h)
            else:
                oh = self.short_size
                if self.fixed_ratio:
                    ow = int(self.short_size * 4.0 / 3.0)
                elif not self.keep_ratio:  # no
                    ow = self.short_size
                else:
                    scale_factor = self.short_size / h
                    oh = int(h * float(scale_factor) +
                             0.5) if self.do_round else int(h *
                                                            self.short_size / w)
                    ow = int(w * float(scale_factor) +
                             0.5) if self.do_round else int(w *
                                                            self.short_size / h)
            if self.backend == 'pillow':
                resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
            elif self.backend == 'cv2' and (self.keep_ratio is not None):
                resized_imgs.append(
                    cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR))
            else:
                resized_imgs.append(
                    Image.fromarray(
                        cv2.resize(np.asarray(img), (ow, oh),
                                   interpolation=cv2.INTER_LINEAR)))
        results['imgs'] = resized_imgs
        return results


@PIPELINES.register()
class RandomCrop(object):
    """
    Random crop images.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, results):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        if 'backend' in results and results['backend'] == 'pyav':  # [c,t,h,w]
            h, w = imgs.shape[2:]
        else:
            w, h = imgs[0].size
        th, tw = self.target_size, self.target_size

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, self.target_size)

        crop_images = []
        if 'backend' in results and results['backend'] == 'pyav':
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            crop_images = imgs[:, :, y1:y1 + th, x1:x1 + tw]  # [C, T, th, tw]
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            for img in imgs:
                if w == tw and h == th:
                    crop_images.append(img)
                else:
                    crop_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = crop_images
        return results


@PIPELINES.register()
class RandomResizedCrop(RandomCrop):
    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 target_size=224,
                 backend='cv2'):

        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.target_size = target_size
        self.backend = backend

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):

        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(np.log(min_ar), np.log(max_ar),
                              size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        imgs = results['imgs']
        if self.backend == 'pillow':
            img_w, img_h = imgs[0].size
        elif self.backend == 'cv2':
            img_h, img_w, _ = imgs[0].shape
        elif self.backend == 'pyav':
            img_h, img_w = imgs.shape[2:]  # [cthw]
        else:
            raise NotImplementedError

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)

        if self.backend == 'pillow':
            img_w, img_h = imgs[0].size
            imgs = [img.crop(left, top, right, bottom) for img in imgs]
        elif self.backend == 'cv2':
            img_h, img_w, _ = imgs[0].shape
            imgs = [img[top:bottom, left:right] for img in imgs]
        elif self.backend == 'pyav':
            img_h, img_w = imgs.shape[2:]  # [cthw]
            imgs = imgs[:, :, top:bottom, left:right]
        else:
            raise NotImplementedError
        results['imgs'] = imgs
        return results


@PIPELINES.register()
class CenterCrop(object):
    """
    Center crop images.
    Args:
        target_size(int): Center crop a square with the target_size from an image.
        do_round(bool): Whether to round up the coordinates of the upper left corner of the cropping area. default: True
    """
    def __init__(self, target_size, do_round=True, backend='pillow'):
        self.target_size = target_size
        self.do_round = do_round
        self.backend = backend

    def __call__(self, results):
        """
        Performs Center crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            ccrop_imgs: List where each item is a PIL.Image after Center crop.
        """
        imgs = results['imgs']
        ccrop_imgs = []
        th, tw = self.target_size, self.target_size
        if isinstance(imgs, paddle.Tensor):
            h, w = imgs.shape[-2:]
            x1 = int(round((w - tw) / 2.0)) if self.do_round else (w - tw) // 2
            y1 = int(round((h - th) / 2.0)) if self.do_round else (h - th) // 2
            ccrop_imgs = imgs[:, :, y1:y1 + th, x1:x1 + tw]
        else:
            for img in imgs:
                if self.backend == 'pillow':
                    w, h = img.size
                elif self.backend == 'cv2':
                    h, w, _ = img.shape
                else:
                    raise NotImplementedError
                assert (w >= self.target_size) and (h >= self.target_size), \
                    "image width({}) and height({}) should be larger than crop size".format(
                        w, h, self.target_size)
                x1 = int(round(
                    (w - tw) / 2.0)) if self.do_round else (w - tw) // 2
                y1 = int(round(
                    (h - th) / 2.0)) if self.do_round else (h - th) // 2
                if self.backend == 'cv2':
                    ccrop_imgs.append(img[y1:y1 + th, x1:x1 + tw])
                elif self.backend == 'pillow':
                    ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        results['imgs'] = ccrop_imgs
        return results


@PIPELINES.register()
class MultiScaleCrop(object):
    """
    Random crop images in with multiscale sizes
    Args:
        target_size(int): Random crop a square with the target_size from an image.
        scales(int): List of candidate cropping scales.
        max_distort(int): Maximum allowable deformation combination distance.
        fix_crop(int): Whether to fix the cutting start point.
        allow_duplication(int): Whether to allow duplicate candidate crop starting points.
        more_fix_crop(int): Whether to allow more cutting starting points.
    """
    def __init__(
            self,
            target_size,  # NOTE: named target size now, but still pass short size in it!
            scales=None,
            max_distort=1,
            fix_crop=True,
            allow_duplication=False,
            more_fix_crop=True,
            backend='pillow'):

        self.target_size = target_size
        self.scales = scales if scales else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.allow_duplication = allow_duplication
        self.more_fix_crop = more_fix_crop
        assert backend in [
            'pillow', 'cv2'
        ], f"MultiScaleCrop's backend must be pillow or cv2, but get {backend}"
        self.backend = backend

    def __call__(self, results):
        """
        Performs MultiScaleCrop operations.
        Args:
            imgs: List where wach item is a PIL.Image.
            XXX:
        results:

        """
        imgs = results['imgs']

        input_size = [self.target_size, self.target_size]

        im_size = imgs[0].size

        # get random crop offset
        def _sample_crop_size(im_size):
            image_w, image_h = im_size[0], im_size[1]

            base_size = min(image_w, image_h)
            crop_sizes = [int(base_size * x) for x in self.scales]
            crop_h = [
                input_size[1] if abs(x - input_size[1]) < 3 else x
                for x in crop_sizes
            ]
            crop_w = [
                input_size[0] if abs(x - input_size[0]) < 3 else x
                for x in crop_sizes
            ]

            pairs = []
            for i, h in enumerate(crop_h):
                for j, w in enumerate(crop_w):
                    if abs(i - j) <= self.max_distort:
                        pairs.append((w, h))
            crop_pair = random.choice(pairs)
            if not self.fix_crop:
                w_offset = random.randint(0, image_w - crop_pair[0])
                h_offset = random.randint(0, image_h - crop_pair[1])
            else:
                w_step = (image_w - crop_pair[0]) / 4
                h_step = (image_h - crop_pair[1]) / 4

                ret = list()
                ret.append((0, 0))  # upper left
                if self.allow_duplication or w_step != 0:
                    ret.append((4 * w_step, 0))  # upper right
                if self.allow_duplication or h_step != 0:
                    ret.append((0, 4 * h_step))  # lower left
                if self.allow_duplication or (h_step != 0 and w_step != 0):
                    ret.append((4 * w_step, 4 * h_step))  # lower right
                if self.allow_duplication or (h_step != 0 or w_step != 0):
                    ret.append((2 * w_step, 2 * h_step))  # center

                if self.more_fix_crop:
                    ret.append((0, 2 * h_step))  # center left
                    ret.append((4 * w_step, 2 * h_step))  # center right
                    ret.append((2 * w_step, 4 * h_step))  # lower center
                    ret.append((2 * w_step, 0 * h_step))  # upper center

                    ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                    ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                    ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                    ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

                w_offset, h_offset = random.choice(ret)

            return crop_pair[0], crop_pair[1], w_offset, h_offset

        crop_w, crop_h, offset_w, offset_h = _sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in imgs
        ]
        if self.backend == 'pillow':
            ret_img_group = [
                img.resize((input_size[0], input_size[1]), Image.BILINEAR)
                for img in crop_img_group
            ]
        else:
            ret_img_group = [
                Image.fromarray(
                    cv2.resize(np.asarray(img),
                               dsize=(input_size[0], input_size[1]),
                               interpolation=cv2.INTER_LINEAR))
                for img in crop_img_group
            ]
        results['imgs'] = ret_img_group
        return results


@PIPELINES.register()
class RandomFlip(object):
    """
    Random Flip images.
    Args:
        p(float): Random flip images with the probability p.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):
        """
        Performs random flip operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            flip_imgs: List where each item is a PIL.Image after random flip.
        """
        imgs = results['imgs']
        v = random.random()
        if v < self.p:
            if isinstance(imgs, paddle.Tensor):
                results['imgs'] = paddle.flip(imgs, axis=[3])
            elif isinstance(imgs[0], np.ndarray):
                results['imgs'] = [cv2.flip(img, 1, img) for img in imgs
                                   ]  # [[h,w,c], [h,w,c], ..., [h,w,c]]
            else:
                results['imgs'] = [
                    img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs
                ]
        else:
            results['imgs'] = imgs
        return results


@PIPELINES.register()
class Image2Array(object):
    """
    transfer PIL.Image to Numpy array and transpose dimensions from 'dhwc' to 'dchw'.
    Args:
        transpose: whether to transpose or not, default True, False for slowfast.
    """
    def __init__(self, transpose=True, data_format='tchw'):
        assert data_format in [
            'tchw', 'cthw'
        ], f"Target format must in ['tchw', 'cthw'], but got {data_format}"
        self.transpose = transpose
        self.data_format = data_format

    def __call__(self, results):
        """
        Performs Image to NumpyArray operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            np_imgs: Numpy array.
        """
        imgs = results['imgs']
        if 'backend' in results and results[
                'backend'] == 'pyav':  # [T,H,W,C] in [0, 1]
            if self.transpose:
                if self.data_format == 'tchw':
                    t_imgs = imgs.transpose((0, 3, 1, 2))  # tchw
                else:
                    t_imgs = imgs.transpose((3, 0, 1, 2))  # cthw
            results['imgs'] = t_imgs
        else:
            t_imgs = np.stack(imgs).astype('float32')
            if self.transpose:
                if self.data_format == 'tchw':
                    t_imgs = t_imgs.transpose(0, 3, 1, 2)  # tchw
                else:
                    t_imgs = t_imgs.transpose(3, 0, 1, 2)  # cthw
            results['imgs'] = t_imgs
        return results


@PIPELINES.register()
class Normalization(object):
    """
    Normalization.
    Args:
        mean(Sequence[float]): mean values of different channels.
        std(Sequence[float]): std values of different channels.
        tensor_shape(list): size of mean, default [3,1,1]. For slowfast, [1,1,1,3]
    """
    def __init__(self, mean, std, tensor_shape=[3, 1, 1], inplace=False):
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}')
        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        self.inplace = inplace
        if not inplace:
            self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
            self.std = np.array(std).reshape(tensor_shape).astype(np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)

    def __call__(self, results):
        """
        Performs normalization operations.
        Args:
            imgs: Numpy array.
        return:
            np_imgs: Numpy array after normalization.
        """
        if self.inplace:
            n = len(results['imgs'])
            h, w, c = results['imgs'][0].shape
            norm_imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(results['imgs']):
                norm_imgs[i] = img

            for img in norm_imgs:  # [n,h,w,c]
                mean = np.float64(self.mean.reshape(1, -1))  # [1, 3]
                stdinv = 1 / np.float64(self.std.reshape(1, -1))  # [1, 3]
                cv2.subtract(img, mean, img)
                cv2.multiply(img, stdinv, img)
        else:
            imgs = results['imgs']
            norm_imgs = imgs / 255.0
            norm_imgs -= self.mean
            norm_imgs /= self.std
            if 'backend' in results and results['backend'] == 'pyav':
                norm_imgs = paddle.to_tensor(norm_imgs, dtype=paddle.float32)
        results['imgs'] = norm_imgs
        return results


@PIPELINES.register()
class JitterScale(object):
    """
    Scale image, while the target short size is randomly select between min_size and max_size.
    Args:
        min_size: Lower bound for random sampler.
        max_size: Higher bound for random sampler.
    """
    def __init__(self,
                 min_size,
                 max_size,
                 short_cycle_factors=[0.5, 0.7071],
                 default_min_size=256):
        self.default_min_size = default_min_size
        self.orig_min_size = self.min_size = min_size
        self.max_size = max_size
        self.short_cycle_factors = short_cycle_factors

    def __call__(self, results):
        """
        Performs jitter resize operations.
        Args:
            imgs (Sequence[PIL.Image]): List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            resized_imgs: List where each item is a PIL.Image after scaling.
        """
        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx in [0, 1]:
            self.min_size = int(
                round(self.short_cycle_factors[short_cycle_idx] *
                      self.default_min_size))
        else:
            self.min_size = self.orig_min_size

        imgs = results['imgs']
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        assert (len(imgs) >= 1), \
            "len(imgs):{} should be larger than 1".format(len(imgs))

        if 'backend' in results and results['backend'] == 'pyav':
            height, width = imgs.shape[2:]
        else:
            width, height = imgs[0].size
        if (width <= height and width == size) or (height <= width
                                                   and height == size):
            return results

        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        if 'backend' in results and results['backend'] == 'pyav':
            frames_resize = F.interpolate(imgs,
                                          size=(new_height, new_width),
                                          mode="bilinear",
                                          align_corners=False)  # [c,t,h,w]
        else:
            frames_resize = []
            for j in range(len(imgs)):
                img = imgs[j]
                scale_img = img.resize((new_width, new_height), Image.BILINEAR)
                frames_resize.append(scale_img)

        results['imgs'] = frames_resize
        return results


@PIPELINES.register()
class MultiCrop(object):
    """
    Random crop image.
    This operation can perform multi-crop during multi-clip test, as in slowfast model.
    Args:
        target_size(int): Random crop a square with the target_size from an image.
    """
    def __init__(self,
                 target_size,
                 default_crop_size=224,
                 short_cycle_factors=[0.5, 0.7071],
                 test_mode=False):
        self.orig_target_size = self.target_size = target_size
        self.short_cycle_factors = short_cycle_factors
        self.default_crop_size = default_crop_size
        self.test_mode = test_mode

    def __call__(self, results):
        """
        Performs random crop operations.
        Args:
            imgs: List where each item is a PIL.Image.
            For example, [PIL.Image0, PIL.Image1, PIL.Image2, ...]
        return:
            crop_imgs: List where each item is a PIL.Image after random crop.
        """
        imgs = results['imgs']
        spatial_sample_index = results['spatial_sample_index']
        spatial_num_clips = results['spatial_num_clips']

        short_cycle_idx = results.get('short_cycle_idx')
        if short_cycle_idx in [0, 1]:
            self.target_size = int(
                round(self.short_cycle_factors[short_cycle_idx] *
                      self.default_crop_size))
        else:
            self.target_size = self.orig_target_size  # use saved value before call

        w, h = imgs[0].size
        if w == self.target_size and h == self.target_size:
            return results

        assert (w >= self.target_size) and (h >= self.target_size), \
            "image width({}) and height({}) should be larger than crop size({},{})".format(w, h, self.target_size, self.target_size)
        frames_crop = []
        if not self.test_mode:
            x_offset = random.randint(0, w - self.target_size)
            y_offset = random.randint(0, h - self.target_size)
        else:  # multi-crop
            x_gap = int(
                math.ceil((w - self.target_size) / (spatial_num_clips - 1)))
            y_gap = int(
                math.ceil((h - self.target_size) / (spatial_num_clips - 1)))
            if h > w:
                x_offset = int(math.ceil((w - self.target_size) / 2))
                if spatial_sample_index == 0:
                    y_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    y_offset = h - self.target_size
                else:
                    y_offset = y_gap * spatial_sample_index
            else:
                y_offset = int(math.ceil((h - self.target_size) / 2))
                if spatial_sample_index == 0:
                    x_offset = 0
                elif spatial_sample_index == spatial_num_clips - 1:
                    x_offset = w - self.target_size
                else:
                    x_offset = x_gap * spatial_sample_index

        for img in imgs:
            nimg = img.crop((x_offset, y_offset, x_offset + self.target_size,
                             y_offset + self.target_size))
            frames_crop.append(nimg)
        results['imgs'] = frames_crop
        return results


@PIPELINES.register()
class PackOutput(object):
    """
    In slowfast model, we want to get slow pathway from fast pathway based on
    alpha factor.
    Args:
        alpha(int): temporal length of fast/slow
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, results):
        fast_pathway = results['imgs']

        # sample num points between start and end
        slow_idx_start = 0
        slow_idx_end = fast_pathway.shape[0] - 1
        slow_idx_num = fast_pathway.shape[0] // self.alpha
        slow_idxs_select = np.linspace(slow_idx_start, slow_idx_end,
                                       slow_idx_num).astype("int64")
        slow_pathway = fast_pathway[slow_idxs_select]

        # T H W C -> C T H W.
        slow_pathway = slow_pathway.transpose(3, 0, 1, 2)
        fast_pathway = fast_pathway.transpose(3, 0, 1, 2)

        # slow + fast
        frames_list = [slow_pathway, fast_pathway]
        results['imgs'] = frames_list
        return results


@PIPELINES.register()
class GroupFullResSample(object):
    def __init__(self, crop_size, flip=False):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (
            crop_size, crop_size)
        self.flip = flip

    def __call__(self, results):
        img_group = results['imgs']

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)

        results['imgs'] = oversample_group
        return results


@PIPELINES.register()
class TenCrop:
    """
    Crop out 5 regions (4 corner points + 1 center point) from the picture,
    and then flip the cropping result to get 10 cropped images, which can make the prediction result more robust.
    Args:
        target_size(int | tuple[int]): (w, h) of target size for crop.
    """
    def __init__(self, target_size):
        self.target_size = (target_size, target_size)

    def __call__(self, results):
        imgs = results['imgs']
        img_w, img_h = imgs[0].size
        crop_w, crop_h = self.target_size
        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4
        offsets = [
            (0, 0),
            (4 * w_step, 0),
            (0, 4 * h_step),
            (4 * w_step, 4 * h_step),
            (2 * w_step, 2 * h_step),
        ]
        img_crops = list()
        for x_offset, y_offset in offsets:
            crop = [
                img.crop(
                    (x_offset, y_offset, x_offset + crop_w, y_offset + crop_h))
                for img in imgs
            ]
            crop_fliped = [
                timg.transpose(Image.FLIP_LEFT_RIGHT) for timg in crop
            ]
            img_crops.extend(crop)
            img_crops.extend(crop_fliped)

        results['imgs'] = img_crops
        return results


@PIPELINES.register()
class UniformCrop:
    """
    Perform uniform spatial sampling on the images,
    select the two ends of the long side and the middle position (left middle right or top middle bottom) 3 regions.
    Args:
        target_size(int | tuple[int]): (w, h) of target size for crop.
    """
    def __init__(self, target_size, backend='cv2'):
        if isinstance(target_size, tuple):
            self.target_size = target_size
        elif isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            raise TypeError(
                f'target_size must be int or tuple[int], but got {type(target_size)}'
            )
        self.backend = backend

    def __call__(self, results):

        imgs = results['imgs']
        if 'backend' in results and results['backend'] == 'pyav':  # [c,t,h,w]
            img_h, img_w = imgs.shape[2:]
        elif self.backend == 'pillow':
            img_w, img_h = imgs[0].size
        else:
            img_h, img_w = imgs[0].shape[:2]

        crop_w, crop_h = self.target_size
        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),
                (w_step * 2, 0),
                (w_step, 0),
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),
                (0, h_step * 2),
                (0, h_step),
            ]
        else:
            raise ValueError(
                f"img_w({img_w}) == crop_w({crop_w}) or img_h({img_h}) == crop_h({crop_h})"
            )
        img_crops = []
        if 'backend' in results and results['backend'] == 'pyav':  # [c,t,h,w]
            for x_offset, y_offset in offsets:
                crop = imgs[:, :, y_offset:y_offset + crop_h,
                            x_offset:x_offset + crop_w]
                img_crops.append(crop)
            img_crops = paddle.concat(img_crops, axis=1)
        else:
            if self.backend == 'pillow':
                for x_offset, y_offset in offsets:
                    crop = [
                        img.crop((x_offset, y_offset, x_offset + crop_w,
                                  y_offset + crop_h)) for img in imgs
                    ]
                    img_crops.extend(crop)
            else:
                for x_offset, y_offset in offsets:
                    crop = [
                        img[y_offset:y_offset + crop_h,
                            x_offset:x_offset + crop_w] for img in imgs
                    ]
                    img_crops.extend(crop)
        results['imgs'] = img_crops
        return results


@PIPELINES.register()
class GroupResize(object):
    def __init__(self, height, width, scale, K, mode='train'):
        self.height = height
        self.width = width
        self.scale = scale
        self.resize = {}
        self.K = np.array(K, dtype=np.float32)
        self.mode = mode
        for i in range(self.scale):
            s = 2**i
            self.resize[i] = paddle.vision.transforms.Resize(
                (self.height // s, self.width // s), interpolation='lanczos')

    def __call__(self, results):
        if self.mode == 'infer':
            imgs = results['imgs']
            for k in list(imgs):  # ("color", 0, -1)
                if "color" in k or "color_n" in k:
                    n, im, _ = k
                    for i in range(self.scale):
                        imgs[(n, im, i)] = self.resize[i](imgs[(n, im, i - 1)])
        else:
            imgs = results['imgs']
            for scale in range(self.scale):
                K = self.K.copy()

                K[0, :] *= self.width // (2**scale)
                K[1, :] *= self.height // (2**scale)

                inv_K = np.linalg.pinv(K)
                imgs[("K", scale)] = K
                imgs[("inv_K", scale)] = inv_K

            for k in list(imgs):
                if "color" in k or "color_n" in k:
                    n, im, i = k
                    for i in range(self.scale):
                        imgs[(n, im, i)] = self.resize[i](imgs[(n, im, i - 1)])

            results['imgs'] = imgs
        return results


@PIPELINES.register()
class ColorJitter(object):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    """
    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 mode='train',
                 p=0.5,
                 keys=None):
        self.mode = mode
        self.colorjitter = paddle.vision.transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, results):
        """
        Args:
            results (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """

        do_color_aug = random.random() > self.p
        imgs = results['imgs']
        for k in list(imgs):
            f = imgs[k]
            if "color" in k or "color_n" in k:
                n, im, i = k
                imgs[(n, im, i)] = f
                if do_color_aug:
                    imgs[(n + "_aug", im, i)] = self.colorjitter(f)
                else:
                    imgs[(n + "_aug", im, i)] = f
        if self.mode == "train":
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]
                del imgs[("color_n", i, -1)]
                del imgs[("color_n_aug", i, -1)]
        else:
            for i in results['frame_idxs']:
                del imgs[("color", i, -1)]
                del imgs[("color_aug", i, -1)]

        results['img'] = imgs
        return results


@PIPELINES.register()
class GroupRandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, results):

        imgs = results['imgs']
        do_flip = random.random() > self.p
        if do_flip:
            for k in list(imgs):
                if "color" in k or "color_n" in k:
                    n, im, i = k
                    imgs[(n, im,
                          i)] = imgs[(n, im,
                                      i)].transpose(Image.FLIP_LEFT_RIGHT)
            if "depth_gt" in imgs:
                imgs['depth_gt'] = np.array(np.fliplr(imgs['depth_gt']))

        results['imgs'] = imgs
        return results


@PIPELINES.register()
class ToArray(object):
    def __init__(self):
        pass

    def __call__(self, results):
        imgs = results['imgs']
        for k in list(imgs):
            if "color" in k or "color_n" in k or "color_aug" in k or "color_n_aug" in k:
                n, im, i = k
                imgs[(n, im,
                      i)] = np.array(imgs[(n, im, i)]).astype('float32') / 255.0
                imgs[(n, im, i)] = imgs[(n, im, i)].transpose((2, 0, 1))
        if "depth_gt" in imgs:
            imgs['depth_gt'] = np.array(imgs['depth_gt']).astype('float32')

        results['imgs'] = imgs
        return results
