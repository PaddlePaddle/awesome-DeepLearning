#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import paddle.nn as nn
import gzip
import json


# # 编写数据导入函数

# In[5]:


def load_data(mode='train'):

    # 数据文件
    datafile = './mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    imgs_length = len(imgs)

    assert len(imgs) == len(labels),           "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# # 编写ResNet网络
# ## 编写残差单元

# In[6]:


class Residual(nn.Layer):
    def __init__(self, num_channels, num_filters, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.use_1x1conv = use_1x1conv
        model = [
            nn.Conv2D(num_channels, num_filters, 3, stride=stride, padding=1),
            nn.BatchNorm2D(num_filters),
            nn.ReLU(),
            nn.Conv2D(num_filters, num_filters, 3, stride=1, padding=1),
            nn.BatchNorm2D(num_filters),
        ]
        self.model = nn.Sequential(*model)
        if use_1x1conv:
            model_1x1 = [nn.Conv2D(num_channels, num_filters, 1, stride=stride)]
            self.model_1x1 = nn.Sequential(*model_1x1)
    def forward(self, X):
        Y = self.model(X)
        if self.use_1x1conv:
            X = self.model_1x1(X)
        return paddle.nn.functional.relu(X + Y)


# ## 编写残差块

# In[7]:


class ResnetBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, num_residuals, first_block=False):
        super(ResnetBlock, self).__init__()
        model = []
        for i in range(num_residuals):
            if i == 0:
                if not first_block:
                    model += [Residual(num_channels, num_filters, use_1x1conv=True, stride=2)]
                else:
                    model += [Residual(num_channels, num_filters)]
            else:
                model += [Residual(num_filters, num_filters)]
        self.model = nn.Sequential(*model)
    def forward(self, X):
        return self.model(X)


# ## ResNet

# In[10]:



class ResNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # ResNet 在输出通道数为64、步幅为2的7×7卷积层后接步幅为2的3×3的最大池化层。每个卷积层后增加的批量归一化层。
        model = [
            nn.Conv2D(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        ]

        # 这里每个模块使用2个残差块
        model += [
            ResnetBlock(64, 64, 2, first_block=True),
            ResnetBlock(64, 128, 2),
            ResnetBlock(128, 256, 2),
            ResnetBlock(256, 512, 2)
        ]

        # 加入全局平均池化层后接上全连接层输出。
        model += [
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Flatten(start_axis=1, stop_axis=-1),
            nn.Linear(512, num_classes),
        ]
        self.model = nn.Sequential(*model)
    def forward(self, X):
        Y = self.model(X)
        return Y


# # 开始训练

# In[9]:


with fluid.dygraph.guard(place=fluid.CUDAPlace(0)):
    model = ResNet()
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
             
            predict = model(image)
            
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')

