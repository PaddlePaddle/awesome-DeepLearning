```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

**损失函数**
	损失函数（loss function）是用来估量模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。
**合页损失函数（hinge loss function）：**
	在机器学习中，hinge loss是一种损失函数，它通常用于"maximum-margin"的分类任务中，如支持向量机。数学表达式为：
	L(y)=max(0,1-\hat{y}y)
   其中 \hat{y} 表示预测输出，通常都是软结果（就是说输出不是0，1这种，可能是0.87。）， y 表示正确的类别。

如果 \hat{y}y<1 ，则损失为： 1-\hat{y}y
如果 \hat{y}y>=1 ，则损失为：0
其函数图像如下，与0-1损失对比：
![](https://ai-studio-static-online.cdn.bcebos.com/b7f8c91439234ab280c15b9685d2272855d23482aa974939b212931f90630ace)

**损失函数python代码实现**
 #svm loss的实现 linear_svm.py
  
 import numpy as np
  rom random import shuffle
  
  def svm_loss_naive(W, X, y, reg):
    """
    用循环实现的SVM loss计算
    这里的loss函数使用的是margin loss
 
   Inputs:
   - W (D, C)： 权重矩阵.
   - X (N, D)： 批输入
   - y (N,) 标签
   - reg: 正则参数
 
   Returns :
   - loss float
   - W的梯度
   """
   dW = np.zeros(W.shape) 
   num_classes = W.shape[1]
   num_train = X.shape[0]
   loss = 0.0
 
   for i in range(num_train):
     scores = X[i].dot(W)
     correct_class_score = scores[y[i]]
     for j in range(num_classes):
       if j == y[i]:
         continue
       margin = scores[j] - correct_class_score + 1 
       if margin > 0:
         loss += margin
         dW[:,j]+=X[i].T
         dW[:,y[i]]-=X[i].T
 
   
   loss /= num_train
   dW/=num_train
   loss += reg * np.sum(W * W)
   dW+=2* reg * W
 
 
   return loss, dW
 
 
 def svm_loss_vectorized(W, X, y, reg):
   """
   不使用循环，利用numpy矩阵运算的特性实现loss和梯度计算
   """
   loss = 0.0
   dW = np.zeros(W.shape) 
 
   #计算loss
   num_classes = W.shape[1]
   num_train = X.shape[0]
   scores=np.dot(X,W)#得到得分矩阵（N，C）
   correct_socre=scores[range(num_train), list(y)].reshape(-1,1)#得到每个输入的正确分类的分数
   margins=np.maximum(0,scores-correct_socre+1)
   margins[range(num_train), list(y)] = 0
   loss=np.sum(margins)/num_train+reg * np.sum(W * W)
   
   #计算梯度
   mask=np.zeros((num_train,num_classes))
   mask[margins>0]=1
   mask[range(num_train),list(y)]-=np.sum(mask,axis=1)
   dW=np.dot(X.T,mask)
   dW/=num_train
   dW+=2* reg * W 
   return loss, dW


**池化方法**
	在卷积神经网络中，我们经常会碰到池化操作，而池化层往往在卷积层后面，通过池化来降低卷积层输出的特征向量，同时改善结果（不易出现过拟合）。
	为什么可以通过降低维度呢？
	因为图像具有一种“静态性”的属性，这也就意味着在一个图像区域有用的特征极有可能在另一个区域同样适用。因此，为了描述大的图像，一个很自然的想法就是对不同位置的特征进行聚合统计，例如，人们可以计算图像一个区域上的某个特定特征的平均值 (或最大值)来代表这个区域的特征。
**空间金字塔池化（Spatial Pyramid Pooling）**
	空间金字塔池化可以把任何尺度的图像的卷积特征转化成相同维度，这不仅可以让CNN处理任意尺度的图像，还能避免cropping和warping操作，导致一些信息的丢失，具有非常重要的意义。
	一般的CNN都需要输入图像的大小是固定的，这是因为全连接层的输入需要固定输入维度，但在卷积操作是没有对图像尺度有限制，所有作者提出了空间金字塔池化，先让图像进行卷积操作，然后转化成维度相同的特征输入到全连接层，这个可以把CNN扩展到任意大小的图像。
   ![](https://ai-studio-static-online.cdn.bcebos.com/f30d81107971498eb253796862023c95f7ecd801eda64fa09bda32d63a4d91a0)
   空间金字塔池化的思想来自于Spatial Pyramid Model，它一个pooling变成了多个scale的pooling。用不同大小池化窗口作用于卷积特征，我们可以得到1X1,2X2,4X4的池化结果，由于conv5中共有256个过滤器，所以得到1个256维的特征，4个256个特征，以及16个256维的特征，然后把这21个256维特征链接起来输入全连接层，通过这种方式把不同大小的图像转化成相同维度的特征。
   ![](https://ai-studio-static-online.cdn.bcebos.com/d43f9d40446b4194a0ebe68f189d08f04bd839e620c44136a5c102e271aca59e)
   对于不同的图像要得到相同大小的pooling结果，就需要根据图像的大小动态的计算池化窗口的大小和步长。假设conv5输出的大小为a*a，需要得到n*n大小的池化结果，可以让窗口大小sizeX为![](https://ai-studio-static-online.cdn.bcebos.com/99a9af01dc314c27a809a7b8f96c1b6f1f1aefb00e4c46d1a5e6ea271dea34f5)，步长为![](https://ai-studio-static-online.cdn.bcebos.com/efa45abf1cc14fe9ad1a9a491f92e26ac39167373153455da4e407ff3d17b494)。
   
   

**数据增强方法**
	1.旋转： 可通过在原图上先放大图像，然后剪切图像得到。
	2.平移：先放大图像，然后水平或垂直偏移位置剪切
	3.缩放：缩放图像
	4.随机遮挡：对图像进行小区域遮挡
	5.水平翻转：以过图像中心的竖直轴为对称轴，将左、右两边像素交换
	6.颜色色差（饱和度、亮度、对比度、 锐度等）
	7.噪声扰动: 对图像的每个像素RGB进行随机扰动, 常用的噪声模式是椒盐噪声和高斯噪声;
   # -*- coding: utf-8 -*-
"""
# 数据增强实现
"""
import tensorflow as tf
import cv2
import numpy as np
from scipy import misc
import random

def random_rotate_image(image):
    interb = ['nearest','bilinear','cubic','bicubic']
    angle = np.random.uniform(low=-10.0, high=10.0)
    key = random.randint(0,3)
    return misc.imrotate(image, angle, interb[key])

def random_occlusion(image):
    b_ratio = 1./10 #遮挡比例
    M1 = np.ones((320,250))
    b_H = random.randint(10,320*(1-b_ratio)-10)  
    b_W = random.randint(10,250*(1-b_ratio)-10)
    M1[b_H:int(b_H+320*b_ratio),b_W:int(b_W+250*b_ratio)] = 0
    M1 = np.expand_dims(M1, 2)
    image = image*M1
    image = image.astype(np.uint8)
    return image

def data_augumrntation(image):
    image = tf.py_func(random_occlusion, [image], tf.uint8) #随机遮挡
    image = tf.py_func(random_rotate_image, [image], tf.uint8) #旋转
    ratio = [0.9,1.1] #缩放比例
    new_H = random.randint(320*ratio[0], 320*ratio[1])
    new_W = random.randint(250*ratio[0], 250*ratio[1])
    print(new_H,new_W)
    image.set_shape((320, 250,3))
    image = tf.image.resize_images(image,[new_H, new_W])
    image = tf.cast(image,tf.uint8)
    image = tf.image.resize_image_with_crop_or_pad(image, 320, 250 )#缩放
    image = tf.random_crop(image, [299, 235, 3]) #随机裁剪
    image = tf.image.random_flip_left_right(image)#镜像
    N_key = random.randint(0,10)
    if N_key == 8:
        image = tf.image.per_image_standardization(image)#标准化
    image = tf.cast(image, tf.float32)
    image = tf.minimum(255.0, tf.maximum(0.0,tf.image.random_brightness(image,25.0)))#光照
    image = tf.minimum(255.0, tf.maximum(0.0,tf.image.random_contrast(image,0.8,1.2)))#对比度
    noise = tf.random_normal((299, 235, 3), mean=0.0, stddev=1.0, dtype=tf.float32)
    image = tf.minimum(255.0, tf.maximum(0.0,image+noise))#随机噪声    
    image = tf.subtract(image,127.5)
    image = tf.multiply(image,0.0078125)    
    return image

if __name__ == '__main__':
    pic = r"bb.jpg"
    file_contents = tf.read_file(pic)
    image = tf.image.decode_jpeg(file_contents, dct_method="INTEGER_ACCURATE")
    R,G,B=tf.unstack(image, num=3, axis=2)
    image=tf.stack([B,G,R], axis=2) #通道转换
    image = data_augumrntation(image)

    #image = tf.cast(image,tf.uint8)
    sess = tf.Session()
    img = sess.run(image)
    cv2.imshow('img',img)
    cv2.waitKey()



**图像分类方法综述**
**图像分类**
	图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。
	图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。
	一般来说，图像分类通过手工特征或特征学习方法对整个图像进行全部描述，然后使用分类器判别物体类别，因此如何提取图像的特征至关重要。
**图像分类常用算法**
**CNN图像分类网络**
**LeNet（1998）**
	网络基本架构为：conv1 (6) -> pool1 -> conv2 (16) -> pool2 -> fc3 (120) -> fc4 (84) -> fc5 (10) -> softmax，括号内数字表示channel数。这是个很小的五层网络（特指卷积或者全连接层），图中subsampling下采样是pooling layer， kernel size 是2x2， stride 2，feature map刚好为上层一半大小。该网络用于对支票（还是邮政？）的手写数字分类。网络受制于当时的硬件条件和训练数据大小，并未带来神经网络的爆发。
![](https://ai-studio-static-online.cdn.bcebos.com/3b7ef02085ef4dd39e9b038b9c2eca7a0c337aa9b3ac47c9a818a4098f8df419)
**AlexNet（2012）**
	AlexNet是2012年ILSVRC（ImageNet Large Scale Visual Recognition Challenge）冠军，以高出10%的正确率力压第二名，这是CNN网络首次获胜，将卷积神经网络的巨大优势带入人们视野。
ILSVRC 历年top5错误率及神经网络深度（层数）：
	AlexNet基本架构为：conv1 (96) -> pool1 -> conv2 (256) -> pool2 -> conv3 (384) -> conv4 (384) -> conv5 (256) -> pool5 -> fc6 (4096) -> fc7 (4096) -> fc8 (1000) -> softmax。AlexNet有着和LeNet相似网络结构，但更深、有更多参数。conv1使用11×11的滤波器、步长为4使空间大小迅速减小(227×227 -> 55×55)。
![](https://ai-studio-static-online.cdn.bcebos.com/c8a9dea0383647ffafc5160ed81fa9ac4ce1ba1a31fe4feb8cbb4a4b0c5f7710)
**VGGNet**
	ILSVRC 2014冠军是GoogLeNet，亚军是VGG。虽然VGG网络是亚军，但是其应用更加广泛。
	VGG网络作者尝试了多种结构，较常用的有VGG16和VGG19（VGG16网络更简单，性能也可以，应用最广泛）。
	VGG16的基本架构为conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3 -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> pool5 -> fc6 (4096) -> fc7 (4096) -> fc8 (1000) -> softmax。 ^3代表重复3次。
	VGG16内存主要消耗在前两层卷积，而参数最主要在第一层全连接中最多。这里说的内存消耗，主要是指存储各层feature map所用的空间，对第一层而言，输入是图片，占用大小就是图片长×宽×通道数，卷积后输出占用的内存就是输出尺寸乘积；参数量中参数是网络需要学习的部分，也就是卷积和全连接层的权重矩阵大小，因为网络中权重矩阵以kernel形式存在，因此参数量就是kernel的（长x宽x通道数）x个数。
![](https://ai-studio-static-online.cdn.bcebos.com/baeb36bffb8a4b619a751e00ec0d77b5ee8b9162e91143748ccfe98b4e7a6b64)



```python
#手写数字识别
import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import numpy as np
from PIL import Image

import gzip
import json

paddle.seed(0)
random.seed(0)
np.random.seed(0)

# 数据文件
datafile = './work/mnist.json.gz'
print('loading mnist dataset from {} ......'.format(datafile))
data = json.load(gzip.open(datafile))
train_set, val_set, eval_set = data

# 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
IMG_ROWS = 28
IMG_COLS = 28
imgs, labels = train_set[0], train_set[1]
print("训练数据集数量: ", len(imgs))
assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
                len(imgs), len(labels))
                
from paddle.io import Dataset

class MnistDataset(Dataset):
    def __init__(self):
        self.IMG_COLS = 28
        self.IMG_ROWS = 28
    def __getitem__(self, idx):
        image = train_set[0][idx]
        image = np.array(image)
        image = image.reshape((1, IMG_ROWS,IMG_COLS)).astype('float32')
        label = train_set[1][idx]
        label = np.array(label)
        label = label.astype('int64')
        return image, label
    def __len__(self):
        return len(imgs)


#调用加载数据的函数
dataset = MnistDataset()
train_loader = paddle.io.DataLoader(dataset, batch_size=100, shuffle=False, return_list=True)

# 定义模型结构
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
     def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         x = F.softmax(x)
         if label is not None:
             acc = paddle.metric.accuracy(input=x, label=label)
             return x, acc
         else:
             return x

use_gpu = False
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

EPOCH_NUM = 5
BATCH_SIZE = 100

paddle.seed(0)

def train(model):

    model.train()

    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(50000//BATCH_SIZE) + 1) * EPOCH_NUM
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data = data[0].reshape([BATCH_SIZE, 1, 28, 28])
            label_data = data[1].reshape([BATCH_SIZE, 1])
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)
            # if batch_id<10:
                # print(label.reshape([-1])[:10])
            #前向计算的过程
            predict, acc = model(image, label)
            avg_acc = paddle.mean(acc)
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predict, label)
            avg_loss = paddle.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

            
    
            # 保存模型参数和优化器的参数
            paddle.save(model.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id)+'.pdparams')
            paddle.save(opt.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id)+'.pdopt')
    print(opt.state_dict().keys())

model = MNIST()
train(model)

print(model.state_dict().keys())

params_path = "./checkpoint/mnist_epoch0"
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
def train_again(model):
    model.train()

    # 读取参数文件
    params_dict = paddle.load(params_path+'.pdparams')
    opt_dict = paddle.load(params_path+'.pdopt')
    # 加载参数到模型
    model.set_state_dict(params_dict)
    
    EPOCH_NUM = 5
    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(50000//BATCH_SIZE) + 1) * EPOCH_NUM
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
    # 使用Adam优化器
    opt = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())
    # 加载参数到优化器
    opt.set_state_dict(opt_dict)

    for epoch_id in range(1, EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data = data[0].reshape([BATCH_SIZE, 1, 28, 28])
            label_data = data[1].reshape([BATCH_SIZE, 1])
            image = paddle.to_tensor(image_data)
            label = paddle.to_tensor(label_data)
            
            #前向计算的过程
            predict, acc = model(image, label)

            avg_acc = paddle.mean(acc)
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predict, label)
            avg_loss = paddle.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
            
            #后向传播，更新参数的过程
            # print(opt.state_dict())
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

model = MNIST()
train_again(model)


```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
