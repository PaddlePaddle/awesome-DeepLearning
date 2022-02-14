"""
tsn frame reader
"""
#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import sys
import random
import functools
import concurrent.futures
import multiprocessing

import numpy as np
import paddle
from PIL import Image, ImageEnhance

from .reader_utils import DataReader


class TSMINFReader(DataReader):
    """
    Data reader for video dataset of jpg folder.
    """

    def __init__(self, name, mode, cfg, material=None):
        super(TSMINFReader, self).__init__(name, mode, cfg)
        name = name.upper()
        self.seg_num        = cfg[name]['seg_num']
        self.seglen         = cfg[name]['seglen']
        self.short_size     = cfg[name]['short_size']
        self.target_size    = cfg[name]['target_size']
        self.batch_size     = cfg[name]['batch_size']
        self.reader_threads = cfg[name]['reader_threads']
        self.buf_size       = cfg[name]['buf_size']
        self.video_path     = cfg[name]['frame_list']

        self.img_mean       = np.array(cfg[name]['image_mean']).reshape([3, 1, 1]).astype(np.float32)
        self.img_std        = np.array(cfg[name]['image_std']).reshape([3, 1, 1]).astype(np.float32)

        self.material = material

    def create_reader(self):
        """
        batch loader for TSN
        """
        _reader = self._inference_reader_creator_longvideo(
                self.video_path,
                self.mode,
                seg_num=self.seg_num,
                seglen=self.seglen,
                short_size=self.short_size,
                target_size=self.target_size,
                img_mean=self.img_mean,
                img_std=self.img_std,
                num_threads = self.reader_threads,
                buf_size = self.buf_size)

        def _batch_reader():
            batch_out = []
            for imgs, label in _reader():
                if imgs is None:
                    continue
                batch_out.append((imgs, label))
                if len(batch_out) == self.batch_size:
                    yield batch_out
                    batch_out = []
            if len(batch_out) > 1:
                yield batch_out[:-1]

        return _batch_reader


    def _inference_reader_creator_longvideo(self, video_path, mode, seg_num, seglen,
                                  short_size, target_size, img_mean, img_std, num_threads, buf_size):
        """
        inference reader for video
        """
        def reader():
            """
            reader
            """
            def image_buf(image_id_path_buf):
                """
                image_buf reader
                """  
                try:
                    img_path = image_id_path_buf[1]
                    img = Image.open(img_path).convert("RGB")
                    image_id_path_buf[2] = img
                except:
                    image_id_path_buf[2] = None

            frame_len = len(video_path)
            read_thread_num = seg_num
            for i in range(0, frame_len, read_thread_num):
                image_list_part = video_path[i: i + read_thread_num]
                image_id_path_buf_list = []
                for k in range(len(image_list_part)):
                    image_id_path_buf_list.append([k, image_list_part[k], None])

                
                with concurrent.futures.ThreadPoolExecutor(max_workers=read_thread_num) as executor:
                    executor.map(lambda image_id_path_buf: image_buf(image_id_path_buf), image_id_path_buf_list)
                imgs_seg_list = [x[2] for x in image_id_path_buf_list]
                    
                # add the fault-tolerant for bad image
                for k in range(len(image_id_path_buf_list)):
                    img_buf = image_id_path_buf_list[k][2]
                    pad_id = 1
                    while pad_id < seg_num and img_buf is None:
                        img_buf = imgs_seg_list[(k + pad_id)%seg_num][2]
                    if img_buf is None:
                        logger.info("read img erro from {} to {}".format(i, i + read_thread_num))
                        exit(0)
                    else:
                        imgs_seg_list[k] = img_buf
                for pad_id in range(len(imgs_seg_list), seg_num):
                    imgs_seg_list.append(imgs_seg_list[-1])
                yield imgs_seg_list      


        def inference_imgs_transform(imgs_list, mode, seg_num, seglen, short_size,\
                                    target_size, img_mean, img_std):
            """
            inference_imgs_transform
            """ 
            imgs_ret = imgs_transform(imgs_list, mode, seg_num, seglen, short_size,
                        target_size, img_mean, img_std)
            label_ret = 0

            return imgs_ret, label_ret

        mapper = functools.partial(
            inference_imgs_transform,
            mode=mode,
            seg_num=seg_num,
            seglen=seglen,
            short_size=short_size,
            target_size=target_size,
            img_mean=img_mean,
            img_std=img_std)

        return paddle.reader.xmap_readers(mapper, reader, num_threads, buf_size, order=True)


def imgs_transform(imgs,
                   mode,
                   seg_num,
                   seglen,
                   short_size,
                   target_size,
                   img_mean,
                   img_std,
                   name=''):
    """
    imgs_transform
    """
    imgs = group_scale(imgs, short_size)

    if mode == 'train':
        if name == "TSM":
            imgs = group_multi_scale_crop(imgs, short_size)
        imgs = group_random_crop(imgs, target_size)
        imgs = group_random_flip(imgs)
    else:
        imgs = group_center_crop(imgs, target_size)

    np_imgs = (np.array(imgs[0]).astype('float32').transpose(
        (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
    for i in range(len(imgs) - 1):
        img = (np.array(imgs[i + 1]).astype('float32').transpose(
            (2, 0, 1))).reshape(1, 3, target_size, target_size) / 255
        np_imgs = np.concatenate((np_imgs, img))
    imgs = np_imgs
    imgs -= img_mean
    imgs /= img_std
    imgs = np.reshape(imgs, (seg_num, seglen * 3, target_size, target_size))

    return imgs

def group_multi_scale_crop(img_group, target_size, scales=None, \
        max_distort=1, fix_crop=True, more_fix_crop=True):
    """
    group_multi_scale_crop
    """
    scales = scales if scales is not None else [1, .875, .75, .66]
    input_size = [target_size, target_size]

    im_size = img_group[0].size

    # get random crop offset
    def _sample_crop_size(im_size):
        """
         _sample_crop_size
        """
        image_w, image_h = im_size[0], im_size[1]

        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in scales]
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
                if abs(i - j) <= max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_step = (image_w - crop_pair[0]) / 4
            h_step = (image_h - crop_pair[1]) / 4

            ret = list()
            ret.append((0, 0))  # upper left
            if w_step != 0:
                ret.append((4 * w_step, 0))  # upper right
            if h_step != 0:
                ret.append((0, 4 * h_step))  # lower left
            if h_step != 0 and w_step != 0:
                ret.append((4 * w_step, 4 * h_step))  # lower right
            if h_step != 0 or w_step != 0:
                ret.append((2 * w_step, 2 * h_step))  # center

            if more_fix_crop:
                ret.append((0, 2 * h_step))  # center left
                ret.append((4 * w_step, 2 * h_step))  # center right
                ret.append((2 * w_step, 4 * h_step))  # lower center
                ret.append((2 * w_step, 0 * h_step))  # upper center

                ret.append((1 * w_step, 1 * h_step))  # upper left quarter
                ret.append((3 * w_step, 1 * h_step))  # upper right quarter
                ret.append((1 * w_step, 3 * h_step))  # lower left quarter
                ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

            w_offset, h_offset = random.choice(ret)
            crop_info = {
                'crop_w': crop_pair[0],
                'crop_h': crop_pair[1],
                'offset_w': w_offset,
                'offset_h': h_offset
                }
             
        return crop_info
    
    crop_info = _sample_crop_size(im_size)
    crop_w = crop_info['crop_w']
    crop_h = crop_info['crop_h']
    offset_w = crop_info['offset_w']
    offset_h = crop_info['offset_h']
    crop_img_group = [
        img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        for img in img_group
    ]
    ret_img_group = [
        img.resize((input_size[0], input_size[1]), Image.BILINEAR)
        for img in crop_img_group
    ]

    return ret_img_group


def group_random_crop(img_group, target_size):
    """
    group_random_crop
    """
    w, h = img_group[0].size
    th, tw = target_size, target_size

    assert (w >= target_size) and (h >= target_size), \
          "image width({}) and height({}) should be larger than crop size".format(w, h)

    out_images = []
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for img in img_group:
        if w == tw and h == th:
            out_images.append(img)
        else:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return out_images


def group_random_flip(img_group):
    """
    group_random_flip
    """
    v = random.random()
    if v < 0.5:
        ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return ret
    else:
        return img_group


def group_center_crop(img_group, target_size):
    """
    group_center_crop
    """
    img_crop = []
    for img in img_group:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
             "image width({}) and height({}) should be larger than crop size".format(w, h)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        img_crop.append(img.crop((x1, y1, x1 + tw, y1 + th)))

    return img_crop


def group_scale(imgs, target_size):
    """
    group_scale
    """
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == target_size) or (h <= w and h == target_size):
            resized_imgs.append(img)
            continue

        if w < h:
            ow = target_size
            oh = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
        else:
            oh = target_size
            ow = int(target_size * 4.0 / 3.0)
            resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))

    return resized_imgs

