#!/usr/bin/env python
# coding: utf-8

# In[19]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[20]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[21]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[22]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[23]:


import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np
import paddle.fluid as fluid


# data包含三个元素的列表：train_set、val_set、 test_set，包括50 000条训练样本、10 000条验证样本、10 000条测试样本。每个样本包含手写数字图片和对应的标签。
# 
# train_set（训练集）：用于确定模型参数。 val_set（验证集）：用于调节模型超参数（如多个网络结构、正则化权重的最优选择）。 test_set（测试集）：用于估计应用效果（没有在模型中应用过的数据，更贴近模型在真实场景应用的效果）。 train_set包含两个元素的列表：train_images、train_labels。
# 
# train_images：[50000,784]的二维列表，包含50 000张图片。每张图片用一个长度为784的向量表示，内容是28*28尺寸的像素灰度值（黑白图片）。 train_labels：[50 000, ]的列表，表示这些图片对应的分类标签，即0~9之间的一个数字。

# In[24]:


# 定义数据集读取器
def load_data(mode='train'):

    # 读取数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels),           "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# 卷积神经网络的结构多种多样，可以在网络的深度上进行延仲，也可在网络的宽度上进行拓展.GoogleNet采用了多个Inception模块来提升网络的深度和宽度，从而达到提高分类准确率，本实验所用的网络是GoogLeNet的简化版。
# 
# 网络中的 Inception模块由4个分支组成，其具体结构如图所示，输入数据分别由4个分支进行处理（处理前后图像尺寸一样)，然后将4个分支的输出堆叠在一起作为下一层的输入。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/2af18e11e26744329bdc2a441e0edbeddbbd5fcd7f3f4ea89ecac4f1b629e0e8)
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/7c44b3fff817434dac21524b9020e4f4b9bf019bdc2d4932b2c96b933547474e)
# 

# In[25]:


class InceptionA(paddle.nn.Layer):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2D(in_channels, 16, kernel_size=1)  

        self.branch5x5_1 = Conv2D(in_channels, 16, kernel_size=1)  
        self.branch5x5_2 = Conv2D(16, 24, kernel_size=5, padding=2)  

        self.branch3x3_1 = Conv2D(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = Conv2D(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = Conv2D(24, 24, kernel_size=3, padding=1)

        self.branch_pool = Conv2D(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]  
        cat = fluid.layers.concat(outputs, axis=1)
        cat = fluid.layers.relu(cat)
        return cat

class Net(paddle.nn.Layer):  
    def __init__(self, name_scope):
        super(Net, self).__init__(name_scope)
        name_scope = self.full_name()
        self.conv1 = Conv2D(1, 10, kernel_size=5, stride=1, padding=0)
        self.conv2 = Conv2D(88, 20, kernel_size=5,stride=1, padding=0)
        
        self.incep1 = InceptionA(in_channels=10)  
        self.incep2 = InceptionA(in_channels=20) 

        self.maxpool = MaxPool2D(kernel_size=2)

        self.fc = Linear(1408, 10)  

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.incep1(x) 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.incep2(x)  
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.softmax(x)
        return x


# In[26]:


model = Net("mnist")

with fluid.dygraph.guard():
    
    
    #调用加载数据的函数
    train_loader = load_data('train')
    #选择优化算法
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        correct = 0
        total = 0
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            
            
            total += label.shape[0]
            pred = predict.argmax(1)
            for i in range(len(pred)):
                if(pred[i] == label[i]):
                    correct += 1
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
        print(correct/total)

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist')


# In[27]:


valid_loader = load_data('eval')

correct = 0
total = 0
for batch_id, data in enumerate(valid_loader()):
     #准备数据，变得更加简洁
    image_data, label_data = data
    image = fluid.dygraph.to_variable(image_data)
    label = fluid.dygraph.to_variable(label_data)
               
    predict = model(image)
                        
    total += label.shape[0]
    pred = predict.argmax(1)
    for i in range(len(pred)):
        if(pred[i] == label[i]):
            correct += 1   
print(correct/total)

