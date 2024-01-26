#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[3]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[4]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[5]:


import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import paddle.nn.functional as F
import random
from paddle.nn import Linear, Conv2D, MaxPool2D
import os
from paddle.io import Dataset
import json
import gzip


# In[6]:


# class MnistDataset(paddle.io.Dataset):
#     def __init__(self, mode):
#         datafile = './work/mnist.json.gz'
#         data = json.load(gzip.open(datafile))
#         # 读取到的数据区分训练集，验证集，测试集
#         train_set, val_set, eval_set = data
#         if mode=='train':
#             # 获得训练数据集
#             imgs, labels = train_set[0], train_set[1]
#         elif mode=='valid':
#             # 获得验证数据集
#             imgs, labels = val_set[0], val_set[1]
#         elif mode=='eval':
#             # 获得测试数据集
#             imgs, labels = eval_set[0], eval_set[1]
#         else:
#             raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
#         # 校验数据
#         imgs_length = len(imgs)
#         assert len(imgs) == len(labels), \
#             "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
#         self.imgs = imgs
#         self.labels = labels

#     def __getitem__(self, idx):
#         img = np.array(self.imgs[idx]).astype('float32')
#         label = np.array(self.labels[idx]).astype('float32')
        
#         return img, label

#     def __len__(self):
#         return len(self.imgs)


# In[ ]:





# In[7]:



# 定义数据集读取器
def load_data(mode='train',BATCHSIZE = 100):
    # 读取数据文件
    datafile = './data/data17168/mnist.json.gz'
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


# In[ ]:





# In[8]:



# 残差模型的建立
#首先创建残差元
class Residual(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

# 然后创建残差块
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

# 最后建立残差网络
class my_resnet(nn.Layer):
    def __init__(self):
        super(my_resnet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2D(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            
            resnet_block(64,64,3,True),
            resnet_block(64, 128, 4),
            resnet_block(128, 256, 2)
        )
        self.avg=nn.AvgPool2D(kernel_size=2)
        self.flatten=nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*2*2, 10)
        )

    def forward(self, x,label=None):
        x = self.net(x)
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.avg(x)
        # x=paddle.reshape(x, [x.shape[0], 64])
        x=self.flatten(x)
        x = self.fc(x)
        x=F.softmax(x,axis=1)
        if label is not None:
             acc = paddle.metric.accuracy(input=x, label=label)
             return x, acc
        else:
             return x


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[9]:


# 这是本次实验用于测试的简单的卷积网络
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs, label=None):
         x = self.conv1(inputs)
         x = F.sigmoid(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.sigmoid(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         x = F.softmax(x)
         if label is not None:
             acc = paddle.metric.accuracy(input=x, label=label)
             return x, acc
         else:
             return x


# In[10]:


# train_dataset = MnistDataset(mode='train')
# # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
# #train_dataset=paddle.vision.datasets.MNIST(mode='train')
# data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)


# In[11]:


#设置随机种子
paddle.seed(0)
#仅优化算法的设置有所差别
def train(model):
    BATCH_SIZE = 100
    EPOCH_NUM = 15
    #开启GPU
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    # device = paddle.device.get_device()
    # paddle.set_device(device)
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train',BATCHSIZE=BATCH_SIZE)
    
    # total_steps = (int(50000//BATCH_SIZE) + 1) * EPOCH_NUM
    # lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    #建立这些用于记录作图
    iter=0
    iters=[]
    losses=[]
    acces=[]
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            #前向计算的过程
            predicts, acc= model(images,labels)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                iters.append(iter)
                losses.append(avg_loss.numpy())
                acces.append(acc.numpy())
                iter = iter + 100
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    # paddle.save(model.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id)+'.pdparams')
    # paddle.save(opt.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id)+'.pdopt')
    # print(opt.state_dict().keys())
    return iters, losses,acces


# In[12]:


#创建模型    
model = my_resnet()
#启动训练过程
iters, losses,acces = train(model)
# print(model.state_dict().keys())


# In[13]:


# 作图训练过程的损失变化
plt.figure()
plt.title("train loss", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(iters, losses,color='red',label='train loss') 
plt.grid()
plt.show()


# In[14]:


# 作图训练过程的训练集准确率变化
plt.figure()
plt.title("train acc", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("acc", fontsize=14)
plt.plot(iters, acces,color='red',label='train acc') 
plt.grid()
plt.show()


# In[15]:


# 对模型的效果使用测试集进行检验
def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    
    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

model = my_resnet()
evaluation(model)

