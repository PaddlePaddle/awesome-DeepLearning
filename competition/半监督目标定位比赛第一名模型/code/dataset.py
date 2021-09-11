import os
import os.path as osp
import cv2
import paddle
import numpy as np
from paddle.io import Dataset
from transform import Normalize, RandomBlur, RandomCrop, RandomHorizontalFlip, RandomVorizontalFlip, Resize, ToTensor, RandomBrightness


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 归一化参数
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        super(Data, self).__init__()
        self.cfg = cfg
        # 下面是数据增强等
        self.randombrig  = RandomBrightness()
        self.normalize   = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop  = RandomCrop()
        self.blur        = RandomBlur()
        self.randomvflip = RandomVorizontalFlip()
        self.randomhflip = RandomHorizontalFlip()
        self.resize      = Resize(384, 384)
        self.totensor    = ToTensor()
        # 读数据
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        # 读取图片
        image = cv2.imread(self.cfg.datapath+'/image/'+name+'.JPEG')[:,:,::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png')[:, :, ::-1].astype(np.float32)
        H, W, C = image.shape
        # 训练的时候的数据增强
        if self.cfg.mode == 'train':
            image, mask = self.randombrig(image, mask)
            image, mask = self.blur(image, mask)
            image, mask = self.normalize(image, mask)
            image, mask = self.randomcrop(image, mask)
            image, mask = self.randomhflip(image, mask)
            image, mask = self.randomvflip(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask
        else:
            # 预测的数据处理
            image, mask = self.normalize(image, mask)
            image, mask = self.resize(image, mask)
            image, mask = self.totensor(image, mask)
            return image, mask, (H, W), name

    def __len__(self):
        return len(self.samples)
