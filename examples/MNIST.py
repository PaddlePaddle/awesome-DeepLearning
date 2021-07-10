#!/usr/bin/env python
# coding: utf-8

# ### MNIST——InceptionV1

# ## 简介
# GoogleNet是google公司推出的高性能网络结构。
# - 其独特的 inception 结构，让网络自行学习不同感受野特征的融合。
# 
# ![inception模块示意图](https://ai-studio-static-online.cdn.bcebos.com/800ec8fdeab14c87a2cfb0fe94ea1bab1294b96ccf164b3aa28bbdbb96460518)
# - 为了应对深度网络训练时，反向传播梯度过小的问题，训练时候将分类 loss 在网络中间引入，加强对浅层网络参数的学习
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/fd0d09e9892f470bb550b4130cb3bc7569ba827d59504dc589094313bfadd6d7)
# 
# 
# - Inception 块由四条并行路径组成。前三条路径使用窗口大小为 11、33 和
# 55 的卷积层，从不同空间大小中提取信息。中间的两条路径在输入上执行 11
# 卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用 33 最大池化
# 层，然后使用 11 卷积层来改变通道数。这四条路径都使用合适的填充来使输入
# 与输出的高和宽一致，最后将每条线路的输出在通道维度上连结，并构成
# Inception 块的输出。在 Inception 块中，通常调整的超参数是每层输出通道的
# 数量。
# Inception 块相当于一个有 4 条路径的子网络。它通过不同窗口形状的卷积
# 层和最大池化层来并行抽取信息，并使用 11 卷积层减少每像素级别上的通道
# 维数从而降低模型复杂度。不同大小的卷积核使得网络可以有效识别不同范围
# 的图像细节，使得网络更加有效。
# ![](https://ai-studio-static-online.cdn.bcebos.com/d059552f5da34fe69de6eab6aeb8b409db4f9eab2c9f46adb2414a152664a304)
# 
# 

# In[1]:


import paddle.fluid as fluid
from paddle.nn import Conv2D, MaxPool2D, Linear
from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np 
from paddle.metric import Accuracy
import random

class InceptionA(paddle.nn.Layer):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch3x3_1 = Conv2D(in_channels, 16, kernel_size=1)
        self.bn1 = fluid.BatchNorm(16)
        self.branch3x3_2 = Conv2D(16, 24, kernel_size=3, padding=1)
        self.bn2 = fluid.BatchNorm(24)
        self.branch3x3_3 = Conv2D(24, 24, kernel_size=3, padding=1)
        self.bn3 = fluid.BatchNorm(24)
        self.branch5x5_1 = Conv2D(in_channels, 16, kernel_size=1)
        self.bn4 = fluid.BatchNorm(16)
        self.branch5x5_2 = Conv2D(16, 24, kernel_size=5, padding=2)
        self.bn5 = fluid.BatchNorm(24)
        self.branch1x1 = Conv2D(in_channels, 16, kernel_size=1)
        self.bn6 = fluid.BatchNorm(16)
        self.branch_pool = Conv2D(in_channels, 24, kernel_size=1)
        self.bn7 = fluid.BatchNorm(24)
    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = F.relu(branch3x3)
        branch3x3 = self.bn1(branch3x3)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = F.relu(branch3x3)
        branch3x3 = self.bn2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch3x3 = F.relu(branch3x3)
        branch3x3 = self.bn3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = F.relu(branch5x5)
        branch5x5 = self.bn4(branch5x5)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = F.relu(branch5x5)
        branch5x5 = self.bn5(branch5x5)

        branch1x1 = self.branch1x1(x)
        branch1x1 = F.relu(branch1x1)
        branch1x1 = self.bn6(branch1x1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_pool = F.relu(branch_pool)
        branch_pool = self.bn7(branch_pool)
        cat = fluid.layers.concat((branch1x1, branch5x5, branch3x3, branch_pool), axis=1)
        output = F.relu(cat)

        return  output

class net(paddle.nn.Layer):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = Conv2D(1, 10, kernel_size=5)  #24*24*10
        self.conv2 = Conv2D(88, 20, kernel_size=5) #8***20
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp =MaxPool2D(2)
        self.fc = Linear(1408, 10)  # 4*4*88=1408

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x =fluid.layers.flatten(x=x, axis=1) 
        x = self.fc(x)

        return x


# In[2]:


#数据处理部分之前的代码，保持不变
import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gzip
import json

# 读取数据
def load_data(mode='train'):
    # 数据预处理，这里用到了随机调整亮度、对比度和饱和度
    transform = Compose([Normalize(mean=[0.1307],std=[0.3081],data_format='CHW')])

#读取训练集 测试集数据
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    print('load finished')
    return train_dataset ,test_dataset
train_dataset ,test_dataset = load_data(mode='train')


# In[3]:


model = paddle.Model(net())  
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()) # adam优化器

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,epochs=2,batch_size=64,verbose=1)
#评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

#训练
def train(model,Batch_size=64):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    model.train()
    iterator = 0
    epochs = 10
    total_steps = (int(50000//Batch_size)+1)*epochs
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01,decay_steps=total_steps,end_lr=0.001)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
                iterator+=200
            optim.step()
            optim.clear_grad()
        paddle.save(model.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdparams')
        paddle.save(optim.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdopt')


#测试
def test(model):
    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    model.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

#随机抽取100张图片进行测试
def random_test(model,num=100):
    select_id = random.sample(range(1, 10000), 100) #生成一百张测试图片的下标
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        label = data[1]
    predicts = model(x_data)
    #返回正确率
    acc = paddle.metric.accuracy(predicts, label)
    print("正确率为：{}".format(acc.numpy()))


if __name__ == '__main__':
    model = net()
    train(model)
    test(model)
    random_test(model)

