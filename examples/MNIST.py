#!/usr/bin/env python
# coding: utf-8

# # 数据集介绍
# 
# MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字

# # 网络模型
# 
# Inception目的是为了增加网络深度和宽度的同时减少参数
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/8176aa00ac44489ea853790a4d1de0ba903232bf968644978b6c224017a9857a)
# 
# 
# Inception v1的亮点：
# 
# 1.卷积层共有的一个功能，可以实现通道方向的降维和增维，至于是降还是增，取决于卷积层的通道数（滤波器个数），在Inception v1中1*1卷积用于降维，减少weights大小和feature map维度。
# 
# 2.1*1卷积特有的功能，由于1*1卷积只有一个参数，相当于对原始feature map做了一个scale，并且这个scale还是训练学出来的，无疑会对识别精度有提升。
# 
# 3.增加了网络的深度
# 
# 4.增加了网络的宽度
# 
# 5.同时使用了1*1，3*3，5*5的卷积，增加了网络对尺度的适应性

# # 优化器选择Adam，损失函数为交叉熵

# In[1]:


from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter

log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

class InceptionA(paddle.nn.Layer):  #作为网络一层
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch3x3_1=paddle.nn.Conv2D(in_channels,16,kernel_size=1) #第一个分支
        self.branch3x3_2=paddle.nn.Conv2D( 16,24,kernel_size=3,padding=1)
        self.branch3x3_3=paddle.nn.Conv2D(24,24,kernel_size=3,padding=1)

        self.branch5x5_1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第二个分支
        self.branch5x5_2=paddle.nn.Conv2D( 16,24,kernel_size=5,padding=2)

        self.branch1x1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第三个分支

        self.branch_pool=paddle.nn.Conv2D(in_channels,24,kernel_size= 1) #第四个分支

    def forward(self,x):
        #分支1处理过程
        branch3x3= self.branch3x3_1(x)
        branch3x3= self.branch3x3_2(branch3x3)
        branch3x3= self.branch3x3_3(branch3x3)
        #分支2处理过程
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        #分支3处理过程
        branch1x1=self.branch1x1(x)
        #分支4处理过程
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding= 1)
        branch_pool=self.branch_pool(branch_pool)
        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]     #将4个分支的输出拼接起来
        return fluid.layers.concat(outputs,axis=1) #横着拼接， 共有24+24+16+24=88个通道

class Net(paddle.nn.Layer):        #卷积，池化，inception，卷积，池化，inception，全连接
    def __init__(self):
        super(Net,self).__init__()
        #定义两个卷积层
        self.conv1=paddle.nn.Conv2D(1,10,kernel_size=5)
        self.conv2=paddle.nn.Conv2D(88,20,kernel_size=5)
        #Inception模块的输出均为88通道
        self.incep1=InceptionA(in_channels=10 )
        self.incep2=InceptionA(in_channels=20)
        self.mp=paddle.nn.MaxPool2D(2)
        self.fc=paddle.nn.Linear(1408,10) #5*5* 88 =2200，图像高*宽*通道数
    def forward(self,x):
        x=F.relu(self.mp(self.conv1(x)))# 卷积池化，relu  输出x为图像尺寸14*14*10
        x =self.incep1(x)               #图像尺寸14*14*88

        x =F.relu(self.mp(self.conv2(x)))# 卷积池化，relu  输出x为图像尺寸5*5*20
        x = self.incep2(x)              #图像尺寸5*5*88

        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.fc(x)
        return x
model = paddle.Model(Net())   # 封装模型
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

