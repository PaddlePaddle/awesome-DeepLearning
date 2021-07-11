#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 引入库
import paddle
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from paddle.vision.transforms import Compose, Normalize
from paddle.metric import Accuracy
import paddle.nn as nn


# In[2]:


# 残差模块
class Residual(nn.Layer):
    def __init__(self, in_channel, out_channel, use_conv1x1=False, stride=1):
        super(Residual, self).__init__()

        # 第一个卷积单元
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu = nn.ReLU()

        # 第二个卷积单元
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(out_channel)

        if use_conv1x1:  # 使用1x1卷积核完成shape匹配,stride=2实现下采样
            self.skip = nn.Conv2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        #  通过 identity 模块
        if self.skip:
            x = self.skip(x)
        #  2 条路径输出直接相加,然后输入激活函数
        output = F.relu(out + x)

        return output

# ResNet18网络模型
class ResNet18_1(nn.Layer):
    # 继承paddle.nn.Layer定义网络结构
    def __init__(self, num_classes=10):
        super(ResNet18_1, self).__init__()
        # 初始化函数(根网络，预处理)
        # x:[b, c, h ,w]=[b,1,28,28]
        self.features = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=7,
                      stride=2, padding=3),  # 第一层卷积,x:[b,64,14,14]
            nn.BatchNorm2D(64),  # 归一化层
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)  # 最大池化，下采样,x:[b,64,56,56]
        )

        # 堆叠 4 个 Block，每个 block 包含了多个残差模块,设置步长不一样
        self.layer1 = build_resblock(64, 64, 2, is_first=True)  # x:[b,64,7,7]
        self.layer2 = build_resblock(64, 128, 2)  # x:[b,128,3,3]
        self.layer3 = build_resblock(128, 256, 2)  # x:[b,256,1,1]
        # self.layer4 = build_resblock(360, 720, 2)  # x:[b,720,7,7]

        # 通过 Pooling 层将高宽降低为 1x1,[b,256,1,1]
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        # 需要拉平为[b,256],不能直接输出连接线性层
        self.flatten = nn.Flatten()
        # 最后连接一个全连接层分类
        self.fc = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, inputs):
        # 前向计算函数：通过根网络
        x = self.features(inputs)
        # 一次通过 4 个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        # 通过池化层
        x = self.avgpool(x)
        # 拉平
        x = self.flatten(x)
        # 通过全连接层
        x = self.fc(x)
        return x


# In[3]:


# 通过build_resblock 可以一次完成2个残差模块的创建。代码如下：
def build_resblock(in_channel, out_channel, num_layers, is_first=False):
    if is_first:
        assert in_channel == out_channel
    block_list = []
    for i in range(num_layers):
        if i == 0 and not is_first:
            block_list.append(Residual(in_channel, out_channel, use_conv1x1=True, stride=2))
        else:
            block_list.append(Residual(out_channel, out_channel))
    resNetBlock = nn.Sequential(*block_list)  # 用*号可以把list列表展开为元素
    return resNetBlock


# In[ ]:


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)


# In[ ]:


# 用Model封装模型
model = paddle.Model(ResNet18_1())  
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
)


# In[ ]:


# 训练模型
model.fit(train_dataset,
          epochs=3,
          batch_size=64,
          verbose=1
          )
# 测试模型
model.evaluate(test_dataset, batch_size=64, verbose=1)

