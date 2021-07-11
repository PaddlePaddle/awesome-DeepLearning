#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入需要的包
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import paddle
print("本教程基于Paddle的版本号为："+paddle.__version__)


# In[2]:


#导入数据集Compose的作用是将用于数据集预处理的接口以列表的方式进行组合。
#导入数据集Normalize的作用是图像归一化处理，支持两种方式： 1. 用统一的均值和标准差值对图像的每个通道进行归一化处理； 2. 对每个通道指定不同的均值和标准差值进行归一化处理。
from paddle.vision.transforms import Compose, Normalize, Resize, RandomRotation, RandomCrop
img_size = 32
#对训练集做数据增强
transform1 = Compose([Resize((img_size+2, img_size+2)), RandomCrop(img_size), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
transform2 = Compose([Resize((img_size, img_size)), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
# 使用transform对数据集做归一化
print('下载并加载训练数据')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform1)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform2)
print('加载完成')


# In[3]:


import paddle
import paddle.nn.functional as F
# 定义多层卷积神经网络
#动态图定义多层卷积神经网络
class ResNet(paddle.nn.Layer):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = paddle.nn.Conv2D(6, 16, 3, padding=1)
        self.conv3 = paddle.nn.Conv2D(16, 32, 3, padding=1)
        self.conv4 = paddle.nn.Conv2D(6, 32, 1)

        self.conv5 = paddle.nn.Conv2D(32, 64, 3, padding=1)
        self.conv6 = paddle.nn.Conv2D(64, 128, 3, padding=1)
        self.conv7 = paddle.nn.Conv2D(32, 128, 1)

        self.maxpool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        
        self.maxpool2 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool3 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool4 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool5 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool6 = paddle.nn.MaxPool2D(2, 2)
        self.flatten=paddle.nn.Flatten()
        self.linear1=paddle.nn.Linear(128, 128)
        self.linear2=paddle.nn.Linear(128, 10)
        self.dropout=paddle.nn.Dropout(0.2)
        self.avgpool=paddle.nn.AdaptiveAvgPool2D(output_size=1)
        
    def forward(self, x):
        y = self.conv1(x)#(bs 6, 32, 32)
        y = F.relu(y)
        y = self.maxpool1(y)#(bs, 6, 16, 16)
        z = y
        y = self.conv2(y)#(bs, 16, 16, 16)
        y = F.relu(y)
        y = self.maxpool2(y)#(bs, 16, 8, 8)
        y = self.conv3(y)#(bs, 32, 8, 8)
        z = self.maxpool4(self.conv4(z))
        y = y+z
        y = F.relu(y)
        z = y
        y = self.conv5(y)#(bs, 64, 8, 8)
        y = F.relu(y)
        y = self.maxpool5(y)#(bs, 64, 4, 4)
        y = self.conv6(y)#(bs, 128, 4, 4)
        z = self.maxpool6(self.conv7(z))
        y = y + z
        y = F.relu(y)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return y


# In[4]:


#定义卷积网络的代码
net_cls = ResNet()
paddle.summary(net_cls, (-1, 1, img_size, img_size))


# In[5]:


from paddle.metric import Accuracy
save_dir = "output/model5_7"
patience = 5
epoch = 20
lr = 0.01
weight_decay = 5e-4
batch_size = 64
momentum = 0.9
# 用Model封装模型
model = paddle.Model(net_cls)

# 定义损失函数
#optim = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=weight_decay)
#lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=10000, eta_min=1e-5)
optim = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters(), momentum=momentum)

visual_dl = paddle.callbacks.VisualDL(log_dir=save_dir)
early_stop = paddle.callbacks.EarlyStopping(monitor='acc', mode='max', patience=patience, 
                                            verbose=1, min_delta=0, baseline=None,
                                            save_best_model=True)
# 配置模型
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())

# 训练保存并验证模型
model.fit(train_dataset,test_dataset,epochs=epoch,batch_size=batch_size,
        save_dir=save_dir,verbose=1, callbacks=[visual_dl, early_stop])


# In[7]:


best_model_path = "output/model5_7/best_model.pdparams"
net_cls = MyNet()
model = paddle.Model(net_cls)
model.load(best_model_path)
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())


# In[ ]:


#用最好的模型在测试集10000张图片上验证
results = model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
print(results)

