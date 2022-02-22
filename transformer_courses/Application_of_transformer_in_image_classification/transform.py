# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import numpy as np
import cv2
from PIL import Image


# 定义decode_image函数，将图片转为Numpy格式
def decode_image(img, to_rgb=True):
    data = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(data, 1)
    if to_rgb:
        assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
        img = img[:, :, ::-1]

    return img


# 定义resize_image函数，对图片大小进行调整
def resize_image(img, size=None, resize_short=None, interpolation=-1):
    interpolation = interpolation if interpolation >= 0 else None
    if resize_short is not None and resize_short > 0:
        resize_short = resize_short
        w = None
        h = None
    elif size is not None:
        resize_short = None
        w = size if type(size) is int else size[0]
        h = size if type(size) is int else size[1]
    else:
        raise ValueError("invalid params for ReisizeImage for '\
            'both 'size' and 'resize_short' are None")

    img_h, img_w = img.shape[:2]
    if resize_short is not None:
        percent = float(resize_short) / min(img_w, img_h)
        w = int(round(img_w * percent))
        h = int(round(img_h * percent))
    else:
        w = w
        h = h
    if interpolation is None:
        return cv2.resize(img, (w, h))
    else:
        return cv2.resize(img, (w, h), interpolation=interpolation)


# 定义crop_image函数，对图片进行裁剪
def crop_image(img, size):
    if type(size) is int:
        size = (size, size)
    else:
        size = size  # (h, w)

    w, h = size
    img_h, img_w = img.shape[:2]
    w_start = (img_w - w) // 2
    h_start = (img_h - h) // 2

    w_end = w_start + w
    h_end = h_start + h
    return img[h_start:h_end, w_start:w_end, :]


# 定义normalize_image函数，对图片进行归一化
def normalize_image(img, scale=None, mean=None, std=None, order=''):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')

    if isinstance(img, Image.Image):
        img = np.array(img)
    assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
    # 对图片进行归一化
    return (img.astype('float32') * scale - mean) / std


# 定义to_CHW_image函数，对图片进行通道变换，将原通道为‘hwc’的图像转为‘chw‘
def to_CHW_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    # 对图片进行通道变换
    return img.transpose((2, 0, 1))


# 图像预处理方法汇总
def transform(data, mode='train'):

    # 图像解码
    data = decode_image(data)
    # 图像缩放
    data = resize_image(data, resize_short=384)
    # 图像裁剪
    data = crop_image(data, size=384)
    # 标准化
    data = normalize_image(
        data, scale=1. / 255., mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # 通道变换
    data = to_CHW_image(data)
    return data
