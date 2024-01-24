#!/usr/bin/python3
# coding=utf-8

import cv2
import numpy as np


# 随机垂直翻转
class RandomVorizontalFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(5) == 1:
            image = image[::-1, :, :].copy()
            mask = mask[::-1, :, :].copy()
        return image.copy(), mask.copy()


# 随机水平翻转
class RandomHorizontalFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
        return image, mask


# 归一化
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        mask /= 255
        return image, mask


# 改尺寸
class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


# 随机对比度
class RandomBrightness(object):
    def __call__(self, image, mask):
        contrast = np.random.rand(1) + 0.5
        light = np.random.randint(-15, 15)
        inp_img = contrast * image + light
        return np.clip(inp_img, 0, 255), mask


# 随机裁剪
class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3, :]


# 随机高斯模糊
class RandomBlur:
    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, im, label):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                im = cv2.GaussianBlur(im, (radius, radius), 0, 0)
        return im, label


# 转为Tensor的格式
class ToTensor(object):
    def __call__(self, image, mask):
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        image, mask = image.astype(np.float32), mask.astype(np.float32)
        mask = mask.mean(axis=0, keepdims=True)
        return image, mask



