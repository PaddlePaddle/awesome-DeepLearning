#!/usr/bin/env python
# coding: utf-8

# 使用内嵌Inception模块的简单卷积神经网络完成MNIST数据集的分类问题。
# Inception模块的核心就是把GoogLeNet的一些大的卷积层换成 1x1, 3x3, 5x5的小卷积层和3x3的池化，一方面增加了网络的宽度，另一方面增加了网络对尺度的适应性。但是5x5卷积层的计算量会非常大，造成特征图的厚度也很大。并且Inception还在3x3，5x5，池化后分别加上1x1的卷积层起到降低特征图厚度的作用，这就是InceptionV1的结构。
# 将InceptionV1放入普通的CNN中作为本次分类所使用的网络。自从AlexNet之后，CNN就被广泛用于图像分类这一领域了。所以本次也使用CNN用来分类

# In[3]:


import paddle
import paddle.nn.functional as F
import paddle.nn as nn
import numpy as np


# In[4]:


from paddle.vision.transforms import Compose,Normalize
transform = Compose([Normalize(mean=[127.5],std=[127.5], data_format='CHW')])
train_set = paddle.vision.datasets.MNIST(mode ='train', transform=transform)
test_set = paddle.vision.datasets.MNIST(mode = 'test', transform=transform)


# In[6]:


class InceptionA(paddle.nn.Layer):

    def __init__(self, in_channles):
        super(InceptionA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=in_channles, out_channels=16, kernel_size=1),
            nn.Conv2D(16, 24, kernel_size=3, padding=1),
            nn.Conv2D(24,24,3,padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channles, 16, 1),
            nn.Conv2D(16, 24, 5,padding=2)
        )
        self.branch3 = nn.Conv2D(in_channles, 16, 1)
        self.branch4 = nn.Conv2D(in_channles, 24, 1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x4 = self.branch4(x4)
        output = [x3, x2, x1, x4]
        return paddle.concat(output, axis=1)
    #Inception模块的定义


# In[14]:


class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(1, 10, 5)
        self.conv2 = nn.Conv2D(88, 20, 5)
        self.incep1 = InceptionA(in_channles=10)
        self.incep2 = InceptionA(in_channles=20)
        self.mp = nn.MaxPool2D(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x))) #12
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x))) #4
        x = self.incep2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc(x)
        return x 
    #带有Inception的简单网络, 结构是卷积-池化-Relu-Inception的两次重复，很简单。


# In[15]:


model = paddle.Model(Net())
loss_func = nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters()) #配置训练用的优化器，损失函数


# In[16]:


from paddle.metric import Accuracy
#根据paddle提供的Model构建实例，使用该API定义训练和测试
#配置模型
model.prepare(
    optimizer,
    loss_func,
    Accuracy()
)
#训练模型
model.fit(train_data=train_set,
    epochs=2,
    batch_size=64,
    verbose=1
)


# In[17]:


model.evaluate(test_set, batch_size=64, verbose=1)
#评价和预测

