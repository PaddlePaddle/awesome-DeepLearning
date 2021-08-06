#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install paddleclas


# In[ ]:


#执行过了就不用再执行了
get_ipython().system('cd work')
get_ipython().system('git clone https://gitee.com/paddlepaddle/PaddleClas/        #从码云下载PaddleClas')
get_ipython().system('cd PaddleClas')

get_ipython().system('pip install --upgrade -r requirements.txt')


# # 1、数据的加载

# In[ ]:


#解压数据集 已经解压到WORK就不用再解压了
get_ipython().system('unzip data/data23828/training.zip -d work/')
get_ipython().system('unzip data/data23828/valid_gt.zip -d work/')
get_ipython().system('unzip data/data23828/validation.zip -d work/')


# In[ ]:


#解压数据集 已经解压到WORK就不用再解压了
get_ipython().system('unzip work/PALM-Training400/PALM-Training400.zip -d work/PALM-Training400/')


# 导入环境

# In[ ]:


import cv2
import random
import numpy as np


# # 2.查看图片形状

# In[ ]:


import os
import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。用在Jupyter notebook中具体作用是当你调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image

DATADIR = 'work/PALM-Training400/PALM-Training400'
# 文件名以N开头的是正常眼底图片，以P开头的是病变眼底图片
file1 = 'N0001.jpg'
file2 = 'P0001.jpg'

# 读取图片
img1 = Image.open(os.path.join(DATADIR, file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR, file2))
img2 = np.array(img2)

# 画出读取的图片
plt.figure(figsize=(16, 8)) # 设置画布
# 位置是由三个整型数值构成，第一个代表行数，第二个代表列数，第三个代表索引位置。举个列子：plt.subplot(2, 3, 5) 和 plt.subplot(235) 是一样一样的。需要注意的是所有的数字不能超过10。
f = plt.subplot(121) 
f.set_title('normal', fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title('PM', fontsize=20)
plt.imshow(img2)
plt.show()


# In[ ]:


# 查看图片形状
img1.shape, img2.shape


# In[ ]:


#数据集中的表格文件居然是xlsx 把他换成csv好做一点
#import pandas as pd 
#import xlrd
#import csv
#data_xls = pd.read_excel('/home/aistudio/work/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx', index_col=0)
#data_xls.to_csv('/home/aistudio/work/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv', encoding='utf-8')


# In[ ]:


#查看csv文件中数据
with open('work/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv') as f:
    datas = f.readlines()
    for data in datas[1:]:
        print(data)
        data = data.split(',')
        filename = data[1]
        label = data[2]
        print(filename)
        print(label)
        break


# # 3.定义数据读取器

# In[ ]:


#设置GPU
device = paddle.CUDAPlace(0)

# 对读入的图像进行预处理 三个处理：变形到[224, 224];图像变为[C,H,W];数据变到[-1,1]之间 
def transform_img(img):
    # 将图片尺寸缩放到224x224
    img = cv2.resize(img, (224, 224))

    # 读入图像尺寸是[H,W,C] 
    # 通过转置操作变为[C,H,W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')  # astype() 数据类型转换

    # 数据变到[-1, 1]之间
    img = img/255
    img -= img*2.0 -1.0
    return img


# 定义训练集数据读取器
def data_loader(datadir, batch_size=30, mode='train', device=device):
    # 将datadir目录下文件列出来，每个文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # H开头的文件名表示高度近似，N开头的文件名表示正常视力
            # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
            if name[0] in ['H', 'N']:
                label = 0
            # P开头的是病理性近视，属于正样本，标签为1
            elif name[0] == 'P':
                label = 1
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就加入到数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)           
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = paddle.to_tensor(batch_imgs, dtype='float32', place=device)
                labels_array = paddle.to_tensor(batch_labels, dtype='int64', place=device).reshape([-1, 1])
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []
        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = paddle.to_tensor(batch_imgs, dtype='float32', place=device)
            labels_array = paddle.to_tensor(batch_labels, dtype='int64', place=device).reshape([-1, 1])
            yield imgs_array, labels_array

    return reader


# 定义验证集数据读取器 验证集的标签是在csv文件中，所以这里和训练集的读取稍有不同
def valid_data_loader(datadir, csvfile, batch_size=30, mode='valid', device=device):
    # 读取csv文件
    datas = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for data in datas[1:]:
            data = data.split(',')
            filename = data[1]
            if filename:
                label = data[2]
                filepath = os.path.join(datadir, filename)
                img = cv2.imread(filepath)
                img = transform_img(img)
                batch_imgs.append(img)
                batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = paddle.to_tensor(batch_imgs, dtype='float32', place=device)
                labels_array = paddle.to_tensor(batch_labels, dtype='int64',place=device).reshape([-1, 1])
                yield imgs_array, labels_array
        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = paddle.to_tensor(batch_imgs, dtype='float32', place=device)
            labels_array = paddle.to_tensor(batch_labels, dtype='int64',place=device).reshape([-1, 1])
            yield imgs_array, labels_array
    return reader


# In[ ]:


# 测试读取器是否顺利运行
datadir = 'work/PALM-Training400/PALM-Training400/'
for batch_imgs, batch_labels in data_loader(datadir)():
    print(batch_imgs.shape)
    print(batch_labels.shape)
    break

datadir = 'work/PALM-Validation400/'
csvfile = 'work/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv'
for batch_imgs, batch_labels in valid_data_loader(datadir,csvfile)():
    print(batch_imgs.shape)
    print(batch_labels.shape)
    break
   


# # 4.构建网络

# ## LeNet的构建
# 
# 虽然咱们是SqueezeNet，不用LeNet，但是了解一下.

# In[ ]:


#看看怎么构建LeNet模型,用paddle.nn.Layer
class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()
        # [3, 224, 224] => [6, 224, 224]
        self.conv1 = nn.Conv2D(3, 6, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # [6, 224, 224] => [6, 112, 112]
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # [6, 112, 112] => [16, 110, 110]
        self.conv2 = nn.Conv2D(6, 16, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU()
        # [16, 110, 110] => [16, 55, 55]
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(16*55*55, 55*55)
        self.fc2 = nn.Linear(55*55, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


# ## SqueezeNet的构建
# 
# 解释一下SqueezeNet
# 论文链接：[https://arxiv.org/pdf/1602.07360.pdf](https://arxiv.org/pdf/1602.07360.pdf)
# 
# SqueezeNet 于2016年11月发布，是一个人工设计的轻量化网络，它在ImageNet上实现了和AlexNet相同水平的正确率，但是只使用了1/50的参数。更进一步，使用模型压缩技术，可以将SqueezeNet压缩到0.5MB，这是AlexNet的1/510。
# 引入了两个术语CNN微结构(microarchitecture)和CNN宏结构(macroarchitecture)。
# 
# CNN微结构： 由层或几个卷积层组成的小模块，如inception模块。
# 
# 
# CNN微结构： 由层或模块组成的完整的网络结构，此时深度是一个重要的参数。
# 
# **SqueezeNet结构设计**
# 
# 网络结构的设计策略：
# 
# （1）代替3x3的滤波器为1x1，这样会减少9倍的参数。
# 
# （2）减少输入到3x3滤波器的输入通道，这样可以进一步减少参数，本文使用squeeze层来实现。
# 
# （3）降采样操作延后，可以给卷积层更大的激活特征图，意味着保留的信息更多，可以提升准确率。
# 
# 
# 策略(1)(2)是减少参数的方案，(3)是在限制参数预算的情况下最大化准确率。
# 

# ### Fire模块
# 
# 
# 作者引入了Fire模块来构造CNN，此模块成功地应用了上述的3个策略。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/08de02b9a8444939ade567f047005b11683e2bfbdf144ec9a84df2d5c46cc77c)
# 
# 
# 
# 
# 模块由squeeze层和expand层组成，squeeze层由1x1的卷积层组成，,可以减少输入expand层的特征图的输入通道。expand层由1x1和3x3的卷积混合而成

# In[ ]:


#paddle.fluid 写法
def fire_module(self, inputs, squeeze_depth, expand_depth, scope):
        with fluid.scope_guard(scope):
            squeeze =fluid.layers.conv2d(inputs, squeeze_depth, filter_size=1,
                                       stride=1, padding="VALID",
                                       act='relu', name="squeeze")
            #print('squeeze shape:',squeeze.shape)
            # squeeze
            expand_1x1 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=1,
                                          stride=1, padding="VALID",
                                          act='relu', name="expand_1x1")
            #print('expand_1x1 shape:',expand_1x1.shape)

            expand_3x3 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=3,
                                          stride=1, padding=1,
                                          act='relu', name="expand_3x3")
            #print('expand_3x3 shape:',expand_3x3.shape)
            return fluid.layers.concat([expand_1x1, expand_3x3], axis=1)


# ### full结构
# 以普通的卷积层(conv1)开始，接着连接8个Fire(2-9)模块，最后以卷积层(conv10)结束。每个Fire模块的filter数量逐渐增加，并且在conv1，Fire4，Fire8和conv10后使用步长为2的max-pooling，这种相对延迟的pooling符合了策略(3)。如下作者对比了添加跳跃层的squeezenet：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/fb125b1c3fe4453ea6d4fd82e66cadf12f6274d3d22c4cc0a2c77631d2efaf6a)
# 
# 
# 其他的实验细节：
# 
# 在3x3的filter之前特征图用0填充1像素边缘，使3x3与1x1卷积的输出具有相同的高度和宽度。
# 
# 在squeeze层和expand层之间使用ReLU函数激励
# 
# Fire9模块之后使用Dropout层，比例为50%。
# 
# 最后不使用全连接层的思想来源于Network in network。
# 
# 
# ### 网络结构维度信息和压缩情况
# 
# 论文使用Han Song提出的Deep Compression对网络进行进一步压缩。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/07df964139ad4e05933b0e2ad61c5942e9e69e7fc05e4ccc86262dc10ca40386)
# 
# 

# In[ ]:


#定义SqueezeNet网络结构:
#paddle.fluid 写法
class SqueezeNet(object):
    def __init__(self, inputs, num_classes=1000):
        # conv_1
        net = fluid.layers.conv2d(inputs, 96, filter_size=7, stride=2,
                                 padding=2, act="relu",
                                 name="conv_1")
        
        # maxpool_1
        net = fluid.layers.pool2d(net, 3, 'max',2,name="maxpool_1")
        
        # fire2
        net = self.fire_module(net, 16, 64, "fire2")
        
        # fire3
        net = self.fire_module(net, 16, 64, "fire3")
        
        # fire4
        net = self.fire_module(net, 32, 128, "fire4")
        
        # maxpool_4
        net = fluid.layers.pool2d(net, pool_size=3, pool_type='max',pool_stride=2,name="maxpool_4")
        
        # fire5
        net = self.fire_module(net, 32, 128, "fire5")
        
        # fire6
        net = self.fire_module(net, 48, 192, "fire6")
        
        # fire7
        net = self.fire_module(net, 48, 192, "fire7")
        
        # fire8
        net = self.fire_module(net, 64, 256, "fire8")
        
        # maxpool_8
        net = fluid.layers.pool2d(net, pool_size=3, pool_type='max',pool_stride=2,name="maxpool_8")
        
        # fire9
        net = self.fire_module(net, 64, 256, "fire9")
        
        # dropout
        net = fluid.layers.dropout(net, 0.5)
        
        # conv_10
        net = fluid.layers.conv2d(net, num_classes, filter_size=1, stride=1,
                               padding="VALID", act=None,
                               name="conv_10")
        net = fluid.layers.batch_norm(net,act='relu')
        # avgpool_10
        net = fluid.layers.pool2d(net, pool_size=13, pool_type='avg',pool_stride=1,name="avgpool_10")
        
        # squeeze the axis
        net = fluid.layers.squeeze(net, axes=[2, 3])
        

        self.logits = net
        self.prediction = fluid.layers.softmax(net)

    def fire_module(self, inputs, squeeze_depth, expand_depth, scope):
        with fluid.scope_guard(scope):
            squeeze =fluid.layers.conv2d(inputs, squeeze_depth, filter_size=1,
                                       stride=1, padding="VALID",
                                       act='relu', name="squeeze")
            #print('squeeze shape:',squeeze.shape)
            # squeeze
            expand_1x1 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=1,
                                          stride=1, padding="VALID",
                                          act='relu', name="expand_1x1")
            #print('expand_1x1 shape:',expand_1x1.shape)

            expand_3x3 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=3,
                                          stride=1, padding=1,
                                          act='relu', name="expand_3x3")
            #print('expand_3x3 shape:',expand_3x3.shape)
            return fluid.layers.concat([expand_1x1, expand_3x3], axis=1)


# In[ ]:


#用pytorch实现squeezenet
#来自 https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py

##首先是fire模块

# class Fire(nn.Module):

#     def __init__(self, inplanes, squeeze_planes,
#                  expand1x1_planes, expand3x3_planes):
#         super(Fire, self).__init__()
#         self.inplanes = inplanes
#         self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
#         self.squeeze_activation = nn.ReLU(inplace=True)
#         self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
#                                    kernel_size=1)
#         self.expand1x1_activation = nn.ReLU(inplace=True)
#         self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
#                                    kernel_size=3, padding=1)
#         self.expand3x3_activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.squeeze_activation(self.squeeze(x))
#         return torch.cat([
#             self.expand1x1_activation(self.expand1x1(x)),
#             self.expand3x3_activation(self.expand3x3(x))
#         ], 1)


# class SqueezeNet(nn.Module):

#     def __init__(self, version=1.0, num_classes=1000):
#         super(SqueezeNet, self).__init__()
#         if version not in [1.0, 1.1]:
#             raise ValueError("Unsupported SqueezeNet version {version}:"
#                              "1.0 or 1.1 expected".format(version=version))
#         self.num_classes = num_classes
#         if version == 1.0:
#             self.features = nn.Sequential(
#                 nn.Conv2d(3, 96, kernel_size=7, stride=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(96, 16, 64, 64),
#                 Fire(128, 16, 64, 64),
#                 Fire(128, 32, 128, 128),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(256, 32, 128, 128),
#                 Fire(256, 48, 192, 192),
#                 Fire(384, 48, 192, 192),
#                 Fire(384, 64, 256, 256),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(512, 64, 256, 256),
#             )
#         else:
#             self.features = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=3, stride=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(64, 16, 64, 64),
#                 Fire(128, 16, 64, 64),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(128, 32, 128, 128),
#                 Fire(256, 32, 128, 128),
#                 nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
#                 Fire(256, 48, 192, 192),
#                 Fire(384, 48, 192, 192),
#                 Fire(384, 64, 256, 256),
#                 Fire(512, 64, 256, 256),
#             )
#         # Final convolution is initialized differently form the rest
#         final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             final_conv,
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m is final_conv:
#                     init.normal_(m.weight, mean=0.0, std=0.01)
#                 else:
#                     init.kaiming_uniform_(m.weight)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x.view(x.size(0), self.num_classes)


# # 训练

# In[ ]:


from paddle.static import InputSpec
from visualdl import LogWriter


# In[ ]:


train_datadir = 'work/PALM-Training400/PALM-Training400/'
epochs = 10
net = LeNet()
train_loader = data_loader(train_datadir, mode='train')
valid_datadir = 'work/PALM-Validation400/'
csvfile = 'work/PALM-Validation-GT/PM_Label_and_Fovea_Location.csv'
valid_loader = valid_data_loader(valid_datadir,csvfile)
modle_path = './model/model'


# In[ ]:


def get_validation_acc(net, valid_loader):
    '''获取测试集acc'''
    net.eval()
    accs = []
    for batch_id, (imgs, labels) in enumerate(valid_loader()):
        # print(imgs.shape)
        out = net(imgs)
        acc = paddle.metric.accuracy(out, labels).numpy()[0]
        accs.append(acc)
    acc = np.array(accs).mean()
    net.train()
    return acc

# tab只能单行缩进
# 多行同时缩进 ctrl + ] ---------小知识

def train(net, epochs, train_loader, valid_loader):
    best_test_acc = 0.962
    net.train()
    loss_fn = nn.CrossEntropyLoss()
    optim = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=0.001)
    iteration = 0

    with LogWriter(logdir='./log/train') as writer:
        for epoch in range(epochs):
            # print('test')
            for batch_id, (imgs, labels) in enumerate(train_loader()):
                
                out = net(imgs)
                loss = loss_fn(out, labels)
                acc = paddle.metric.accuracy(out, labels)
                
                if batch_id % 5 == 0:
                    test_acc = get_validation_acc(net, valid_loader)
                    # 加入visualDL可视化
                    iteration += 1
                    writer.add_scalar(tag='acc', step=iteration, value=acc)
                    writer.add_scalar(tag='test_acc', step=iteration, value=test_acc)
                    
                    print('Epoch{}/{} iter{} loss={} acc={} test_acc={}'.format(epochs, epoch, batch_id, loss.numpy()[0], acc.numpy()[0], test_acc))
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    paddle.jit.save(
                        layer = net,
                        path = modle_path,
                        input_spec = [InputSpec(shape=[None, 3, 224, 224], dtype='float32')]
                    )
                    print('最佳模型已保存, 最佳epoch{} iter{} best_test_acc={}'.format(epoch, batch_id, best_test_acc))

                loss.backward()
                optim.step()
                optim.clear_grad()


# In[ ]:


train(net, 5, train_loader, valid_loader)


# In[ ]:


# 加载模型并计算测试集准确率
path = './model/model'
loaded_layer = paddle.jit.load(path)
loaded_layer.eval()

def evaluation(net, valid_loader):
    accs = []
    for batch_id, (imgs, labels) in enumerate(valid_loader()):
        # print(imgs.shape)
        out = net(imgs)
        acc = paddle.metric.accuracy(out, labels).numpy()[0]
        accs.append(acc)
    acc = np.array(accs).mean()
    return acc

acc = evaluation(loaded_layer, valid_loader)
print(acc)


# 参考链接：
# 
# iChallenge-PM 数据集说明： [https://ai.baidu.com/broad/introduction](https://ai.baidu.com/broad/introduction)
# 
# 公开眼疾数据集: [https://aistudio.baidu.com/aistudio/datasetdetail/23828/1](https://aistudio.baidu.com/aistudio/datasetdetail/23828/1)
# 
# aistudio 眼疾识别案例深度实践计算机视觉 ppt：[https://aistudio.baidu.com/aistudio/education/preview/1595640](https://aistudio.baidu.com/aistudio/education/preview/1595640)
# 
# SqueezeNet论文链接：[https://arxiv.org/pdf/1602.07360.pdf](https://arxiv.org/pdf/1602.07360.pdf)
# 
# paddle API文档：[https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)
# 
