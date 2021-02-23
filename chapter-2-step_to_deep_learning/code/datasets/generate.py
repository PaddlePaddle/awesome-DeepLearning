# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# coding=utf-8

import numpy as np
import json
import gzip
import random
import paddle


def load_data(mode='train', batch_size=64):
    datafile = './mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    # 加载json数据文件
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')

    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, test_set = data
    if mode == 'train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'val':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'test':
        # 获得测试数据集
        imgs, labels = test_set[0], test_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    print("训练数据集数量: ", len(imgs))

    # 校验数据
    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))

    # 获得数据集长度
    imgs_length = len(imgs)

    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成希望的类型
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list) == batch_size:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './datasets/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, test_set = data
        if mode == 'train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode == 'val':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode == 'test':
            # 获得测试数据集
            imgs, labels = test_set[0], test_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")

        # 校验数据
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))

        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx]).astype('float32')
        label = np.array(self.labels[idx]).astype('int64')

        return img, label

    def __len__(self):
        return len(self.imgs)
