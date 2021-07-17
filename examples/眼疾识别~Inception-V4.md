# 1、理论知识

&emsp;&emsp;卷积网络在图像识别领域已经十分流行，经典网络有AlexNet、VGGNet、GoogLeNet等。Residual connection的提出是用于训练更深的网络，但是作者发现不使用Residual connection也可以训练更深的网络。Residual connection并不是必要条件，只是使用了Residual connection会加快训练速度。Inception结构最初由GoogLeNet引入，GoogLeNet叫做Inception-v1；之后引入了BatchNormalization，叫做Inception-v2；随后引入分解，叫做Inception-v3。

&emsp;&emsp;Inception结构有着良好的性能，且计算量低。Residual connection不同于传统网络结构，且在2015 ILSVRC取得冠军，它的性能和Inception-v3接近。作者尝试将Inception结构和Residual connection结合，同时也设计了不用Residual connection版本的Inception-v4。

## 1.1Inception进化史

### Inception v1

&emsp;&emsp;Inception v1即大名鼎鼎的GoogLeNet，Google在2014年ImageNet比赛中夺冠的大杀器。相比之前的AlexNet和ZFNet，Inception v1在结构上有两个突出的特点：
1. Multi-branch结构。每个inception module分为四个branch，输出时对所有branch的输出做concatenation
1. Heterogeneous branch结构。每个branch的结构都是不一样的，主要表现在branch的depth和kernel size两方面。

### Inception v2、V3

&emsp;&emsp;Inception  v2和v3是在同一篇文章中提出来的。相比Inception v1，结构上的改变主要有两点：1）用堆叠的小kernel  size（3*3）的卷积来替代Inception v1中的大kernel size（5*5）卷积；2）引入了空间分离卷积（Factorized  Convolution）来进一步降低网络的复杂度。

###   Inception v4

&emsp;&emsp;Inception  v4在模块设计上并没有提出非常有insight的princple，感觉只是把之前的Inception  module变得更复杂了，然后对网络的stem进行了一定的修改。Inception-ResNet在Inception  module中引入了residual connection。

## 1.2 Inception的主流解释

&emsp;&emsp;Inception系列的结构框架在v1的时候就基本确定下来的，后续的版本是对v1上的迭代升级，而非颠覆。所以，大家关于Inception的解读也都侧重于v1。总的来说，大家普遍认为Inception module的两个设计原则是它成功的关键：

1. Multi-branch结构。Inception  v1的作者受人脑结构的启发，认为CNN内部的连接应该具有一定的稀疏性，所以在Inception  module中引入了multi-branch的结构。因为每个branch内部1*1conv的应用，使得Inception  module相比single branch结构的FLOPs要低。
1. Heterogeneous  branch结构。Inception中每个branch的结构都是不一样的，特别是kernel  size。作者认为同类别的对象在不同图片中的size可能是不同的，所以应该在同一个module内部融合不同size的kernel产生的feature，有利于CNN识别不同size的对象。


# 2、网络结构
## 2.1Inception v4中的三个基本模块
![](https://ai-studio-static-online.cdn.bcebos.com/da40a93f99f64a919bd0c22b9cba08be7dc4e717d2184aa09cc22db9336b820c)
1. 左图是基本的Inception v2/v3模块，使用两个3x3卷积代替5x5卷积，并且使用average pooling，该模块主要处理尺寸为35x35的feature map；
1. 中图模块使用1xn和nx1卷积代替nxn卷积，同样使用average pooling，该模块主要处理尺寸为17x17的feature map；
1. 右图在原始的8x8处理模块上将3x3卷积用1x3卷积和3x1卷积。 

## 2.2 Inception v4网络结构(右图为Inception v4的Stem模块)
![](https://ai-studio-static-online.cdn.bcebos.com/46040c6650914a0684753a880e203b974344bde9f7304fd3b362ba05d359ba53)

## 2.3 整体网络结构
![](https://ai-studio-static-online.cdn.bcebos.com/4028f8cb075a43eab71e327d5ebf674bc262170f6bae413f8eacff64742c3015)



# 3、代码实现
## 数据可视化


```python
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
file1 = 'N0011.jpg'
file2 = 'P0085.jpg'
# 读取图片
img1 = Image.open(os.path.join(DATADIR, file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR, file2))
img2 = np.array(img2)
# 画出读取的图片
#plt.figure(figsize=(16, 8))
#f = plt.subplot(121)
#f.set_title('Normal', fontsize=20)
#plt.imshow(img1)
#f = plt.subplot(122)
#f.set_title('PM', fontsize=20)
#plt.imshow(img2)
#plt.show()
```

![](https://ai-studio-static-online.cdn.bcebos.com/7085ef06732f4ebdaddc5d9fa5b2a0a356ab7335347c4654812ebcc2b7e15118)



```python
#定义数据读取器
import cv2
import random
import numpy as np

# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


```


```python
# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader


```


```python
# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            # 根据图片文件名加载图片，并对图像数据作预处理
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader
```


```python
# 查看数据形状
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
train_loader = data_loader(DATADIR, 
                           batch_size=10, mode='train')
data_reader = train_loader()
data = next(data_reader)
data[0].shape, data[1].shape
```




    ((10, 3, 224, 224), (10, 1))




```python
!pip install xlrd==1.2.0
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Collecting xlrd==1.2.0
    [?25l  Downloading https://mirrors.bfsu.edu.cn/pypi/web/packages/b0/16/63576a1a001752e34bf8ea62e367997530dc553b689356b9879339cf45a4/xlrd-1.2.0-py2.py3-none-any.whl (103kB)
    [K     |████████████████████████████████| 112kB 7.2MB/s eta 0:00:01
    [?25hInstalling collected packages: xlrd
    Successfully installed xlrd-1.2.0



```python
import pandas as pd
df=pd.read_excel('/home/aistudio/work/palm/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx')
df.to_csv('/home/aistudio/work/palm/PALM-Validation-GT/labels.csv',index=False)
```

## 训练和评估代码


```python
#训练和评估代码
import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np

DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/work/palm/PALM-Validation400'
CSVFILE = '/home/aistudio/work/palm/PALM-Validation-GT/labels.csv'

# 定义训练过程
def train(model):
    with fluid.dygraph.guard():
        print('start training ... ')
        model.train()
        epoch_num = 5
        # 定义优化器
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        # 定义数据读取器，训练数据读取器和验证数据读取器
        train_loader = data_loader(DATADIR, batch_size=10, mode='train')
        valid_loader = valid_data_loader(DATADIR2, CSVFILE)
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                label = fluid.layers.cast(label, dtype='int64')
                one_hot_label = fluid.layers.one_hot(input=label, depth=2)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                # 进行loss计算
                loss = fluid.layers.cross_entropy(logits, label)
                avg_loss = fluid.layers.mean(loss)

                if batch_id % 10 == 0:
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                label = fluid.layers.cast(label, dtype='int64')
                one_hot_label = fluid.layers.one_hot(input=label, depth=2)
                # 运行模型前向计算，得到预测值
                logits = model(img)
                # 进行loss计算
                loss = fluid.layers.cross_entropy(logits, label)
                
                acc = fluid.layers.accuracy(logits, fluid.layers.cast(label, dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            model.train()
```

## Inception V4 网络搭建



```python
# -*- coding:utf-8 -*-

# InceptionV4模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, FC
from paddle.fluid.dygraph.base import to_variable

import os
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
import logging

from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance

class InceptionV4(fluid.dygraph.Layer):

    def __init__(self,name_scope):
        super(InceptionV4,self).__init__(name_scope)
        pass

    
    def name(self):
        """
        返回网络名字
        :return:
        """
        return 'InceptionV4'

    def forward(self, input, class_dim=2):
        x = self.inception_stem(input)

        for i in range(4):
            x = self.inceptionA(x, name=str(i + 1))
        x = self.reductionA(x)

        for i in range(7):
            x = self.inceptionB(x, name=str(i + 1))
        x = self.reductionB(x)

        for i in range(3):
            x = self.inceptionC(x, name=str(i + 1))

        pool = fluid.layers.pool2d(
            input=x, pool_size=8, pool_type='avg', global_pooling=True)

        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)

        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            act='softmax',
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="final_fc_weights"),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="final_fc_offset"))
        return out

    def conv_bn_layer(self,
                      data,
                      num_filters,
                      filter_size,
                      stride=1,
                      padding=0,
                      groups=1,
                      act='relu',
                      name=None):
        conv = fluid.layers.conv2d(
            input=data,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name)
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def inception_stem(self, data, name=None):
        conv = self.conv_bn_layer(
            data, 32, 3, stride=2, act='relu', name="conv1_3x3_s2")
        conv = self.conv_bn_layer(conv, 32, 3, act='relu', name="conv2_3x3_s1")
        conv = self.conv_bn_layer(
            conv, 64, 3, padding=1, act='relu', name="conv3_3x3_s1")

        pool1 = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_stride=2, pool_type='max')
        conv2 = self.conv_bn_layer(
            conv, 96, 3, stride=2, act='relu', name="inception_stem1_3x3_s2")
        concat = fluid.layers.concat([pool1, conv2], axis=1)

        conv1 = self.conv_bn_layer(
            concat, 64, 1, act='relu', name="inception_stem2_3x3_reduce")
        conv1 = self.conv_bn_layer(
            conv1, 96, 3, act='relu', name="inception_stem2_3x3")

        conv2 = self.conv_bn_layer(
            concat, 64, 1, act='relu', name="inception_stem2_1x7_reduce")
        conv2 = self.conv_bn_layer(
            conv2,
            64, (7, 1),
            padding=(3, 0),
            act='relu',
            name="inception_stem2_1x7")
        conv2 = self.conv_bn_layer(
            conv2,
            64, (1, 7),
            padding=(0, 3),
            act='relu',
            name="inception_stem2_7x1")
        conv2 = self.conv_bn_layer(
            conv2, 96, 3, act='relu', name="inception_stem2_3x3_2")

        concat = fluid.layers.concat([conv1, conv2], axis=1)

        conv1 = self.conv_bn_layer(
            concat, 192, 3, stride=2, act='relu', name="inception_stem3_3x3_s2")
        pool1 = fluid.layers.pool2d(
            input=concat, pool_size=3, pool_stride=2, pool_type='max')

        concat = fluid.layers.concat([conv1, pool1], axis=1)

        return concat

    def inceptionA(self, data, name=None):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(
            pool1, 96, 1, act='relu', name="inception_a" + name + "_1x1")

        conv2 = self.conv_bn_layer(
            data, 96, 1, act='relu', name="inception_a" + name + "_1x1_2")

        conv3 = self.conv_bn_layer(
            data, 64, 1, act='relu', name="inception_a" + name + "_3x3_reduce")
        conv3 = self.conv_bn_layer(
            conv3,
            96,
            3,
            padding=1,
            act='relu',
            name="inception_a" + name + "_3x3")

        conv4 = self.conv_bn_layer(
            data,
            64,
            1,
            act='relu',
            name="inception_a" + name + "_3x3_2_reduce")
        conv4 = self.conv_bn_layer(
            conv4,
            96,
            3,
            padding=1,
            act='relu',
            name="inception_a" + name + "_3x3_2")
        conv4 = self.conv_bn_layer(
            conv4,
            96,
            3,
            padding=1,
            act='relu',
            name="inception_a" + name + "_3x3_3")

        concat = fluid.layers.concat([conv1, conv2, conv3, conv4], axis=1)

        return concat

    def reductionA(self, data, name=None):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_stride=2, pool_type='max')

        conv2 = self.conv_bn_layer(
            data, 384, 3, stride=2, act='relu', name="reduction_a_3x3")

        conv3 = self.conv_bn_layer(
            data, 192, 1, act='relu', name="reduction_a_3x3_2_reduce")
        conv3 = self.conv_bn_layer(
            conv3, 224, 3, padding=1, act='relu', name="reduction_a_3x3_2")
        conv3 = self.conv_bn_layer(
            conv3, 256, 3, stride=2, act='relu', name="reduction_a_3x3_3")

        concat = fluid.layers.concat([pool1, conv2, conv3], axis=1)

        return concat

    def inceptionB(self, data, name=None):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(
            pool1, 128, 1, act='relu', name="inception_b" + name + "_1x1")

        conv2 = self.conv_bn_layer(
            data, 384, 1, act='relu', name="inception_b" + name + "_1x1_2")

        conv3 = self.conv_bn_layer(
            data, 192, 1, act='relu', name="inception_b" + name + "_1x7_reduce")
        conv3 = self.conv_bn_layer(
            conv3,
            224, (1, 7),
            padding=(0, 3),
            act='relu',
            name="inception_b" + name + "_1x7")
        conv3 = self.conv_bn_layer(
            conv3,
            256, (7, 1),
            padding=(3, 0),
            act='relu',
            name="inception_b" + name + "_7x1")

        conv4 = self.conv_bn_layer(
            data,
            192,
            1,
            act='relu',
            name="inception_b" + name + "_7x1_2_reduce")
        conv4 = self.conv_bn_layer(
            conv4,
            192, (1, 7),
            padding=(0, 3),
            act='relu',
            name="inception_b" + name + "_1x7_2")
        conv4 = self.conv_bn_layer(
            conv4,
            224, (7, 1),
            padding=(3, 0),
            act='relu',
            name="inception_b" + name + "_7x1_2")
        conv4 = self.conv_bn_layer(
            conv4,
            224, (1, 7),
            padding=(0, 3),
            act='relu',
            name="inception_b" + name + "_1x7_3")
        conv4 = self.conv_bn_layer(
            conv4,
            256, (7, 1),
            padding=(3, 0),
            act='relu',
            name="inception_b" + name + "_7x1_3")

        concat = fluid.layers.concat([conv1, conv2, conv3, conv4], axis=1)

        return concat

    def reductionB(self, data, name=None):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_stride=2, pool_type='max')

        conv2 = self.conv_bn_layer(
            data, 192, 1, act='relu', name="reduction_b_3x3_reduce")
        conv2 = self.conv_bn_layer(
            conv2, 192, 3, stride=2, act='relu', name="reduction_b_3x3")

        conv3 = self.conv_bn_layer(
            data, 256, 1, act='relu', name="reduction_b_1x7_reduce")
        conv3 = self.conv_bn_layer(
            conv3,
            256, (1, 7),
            padding=(0, 3),
            act='relu',
            name="reduction_b_1x7")
        conv3 = self.conv_bn_layer(
            conv3,
            320, (7, 1),
            padding=(3, 0),
            act='relu',
            name="reduction_b_7x1")
        conv3 = self.conv_bn_layer(
            conv3, 320, 3, stride=2, act='relu', name="reduction_b_3x3_2")

        concat = fluid.layers.concat([pool1, conv2, conv3], axis=1)

        return concat

    def inceptionC(self, data, name=None):
        pool1 = fluid.layers.pool2d(
            input=data, pool_size=3, pool_padding=1, pool_type='avg')
        conv1 = self.conv_bn_layer(
            pool1, 256, 1, act='relu', name="inception_c" + name + "_1x1")

        conv2 = self.conv_bn_layer(
            data, 256, 1, act='relu', name="inception_c" + name + "_1x1_2")

        conv3 = self.conv_bn_layer(
            data, 384, 1, act='relu', name="inception_c" + name + "_1x1_3")
        conv3_1 = self.conv_bn_layer(
            conv3,
            256, (1, 3),
            padding=(0, 1),
            act='relu',
            name="inception_c" + name + "_1x3")
        conv3_2 = self.conv_bn_layer(
            conv3,
            256, (3, 1),
            padding=(1, 0),
            act='relu',
            name="inception_c" + name + "_3x1")

        conv4 = self.conv_bn_layer(
            data, 384, 1, act='relu', name="inception_c" + name + "_1x1_4")
        conv4 = self.conv_bn_layer(
            conv4,
            448, (1, 3),
            padding=(0, 1),
            act='relu',
            name="inception_c" + name + "_1x3_2")
        conv4 = self.conv_bn_layer(
            conv4,
            512, (3, 1),
            padding=(1, 0),
            act='relu',
            name="inception_c" + name + "_3x1_2")
        conv4_1 = self.conv_bn_layer(
            conv4,
            256, (1, 3),
            padding=(0, 1),
            act='relu',
            name="inception_c" + name + "_1x3_3")
        conv4_2 = self.conv_bn_layer(
            conv4,
            256, (3, 1),
            padding=(1, 0),
            act='relu',
            name="inception_c" + name + "_3x1_3")

        concat = fluid.layers.concat(
            [conv1, conv2, conv3_1, conv3_2, conv4_1, conv4_2], axis=1)

        return concat
```

## 开始训练


```python
with fluid.dygraph.guard():
    model = InceptionV4('InceptionV4')

train(model)
```

    start training ... 
    epoch: 0, batch_id: 0, loss is: [0.88028544]
    epoch: 0, batch_id: 10, loss is: [0.6625659]
    epoch: 0, batch_id: 20, loss is: [0.77900946]
    epoch: 0, batch_id: 30, loss is: [0.81022865]
    [validation] accuracy/loss: 0.4350000023841858/0.7083271741867065
    epoch: 1, batch_id: 0, loss is: [0.69155926]
    epoch: 1, batch_id: 10, loss is: [0.6982287]
    epoch: 1, batch_id: 20, loss is: [0.85588396]
    epoch: 1, batch_id: 30, loss is: [0.56334096]
    [validation] accuracy/loss: 0.4949999749660492/0.6976350545883179
    epoch: 2, batch_id: 0, loss is: [0.7289728]
    epoch: 2, batch_id: 10, loss is: [0.79815567]
    epoch: 2, batch_id: 20, loss is: [0.76212335]
    epoch: 2, batch_id: 30, loss is: [0.7391117]
    [validation] accuracy/loss: 0.5550000071525574/0.6830703616142273
    epoch: 3, batch_id: 0, loss is: [0.61926126]
    epoch: 3, batch_id: 10, loss is: [0.6321007]
    epoch: 3, batch_id: 20, loss is: [0.68893206]
    epoch: 3, batch_id: 30, loss is: [0.6742482]
    [validation] accuracy/loss: 0.5099999904632568/0.6953158378601074
    epoch: 4, batch_id: 0, loss is: [0.7741206]
    epoch: 4, batch_id: 10, loss is: [0.69841546]
    epoch: 4, batch_id: 20, loss is: [0.79569894]
    epoch: 4, batch_id: 30, loss is: [0.7748053]
    [validation] accuracy/loss: 0.5175000429153442/0.6993484497070312

