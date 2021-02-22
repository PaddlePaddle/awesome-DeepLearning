#!/usr/bin/env python
# coding: utf-8

'''
 定义 SimpleNet 网络结构
'''

import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

class SimpleNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        #super(SimpleNet, self).__init__(name_scope)
        self.conv1 = Conv2D(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, tride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.max_pool2 = MaxPool2D(kernel_size=2, tride=2)
        self.fc1 = Linear(in_features=50176, out_features=64)
        self.fc2 = Linear(in_features=64, out_features=num_classes)        

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
