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


# In[ ]:





# In[5]:


import random
import os
import cv2
import numpy as np
import math
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import paddle.nn as nn
import matplotlib.pyplot as plt
import paddle.nn.functional as F
from paddle.fluid.param_attr import ParamAttr


# In[6]:


# !unzip -oq /home/aistudio/PALM-Training400/PALM-Training400.zip -d /home/aistudio/PALM-Training400/
# !unzip -oq /home/aistudio/data/data31387/validation.zip 


# In[7]:


def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224,224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


# In[ ]:


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


# In[ ]:


# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47
    # 打开包含验证集标签的csvfile，并读入其中的内容
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


# In[ ]:


class LeNet(nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()

        # 创建卷积和池化层块，每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        self.conv1 = Conv2D(num_channels=3, num_filters=6, filter_size=5, act='sigmoid')
        self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = Conv2D(num_channels=6, num_filters=16, filter_size=5, act='sigmoid')
        self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        # 创建第3个卷积层
        self.conv3 = Conv2D(num_channels=16, num_filters=120, filter_size=4, act='sigmoid')
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc1 = Linear(input_dim=300000, output_dim=64, act='sigmoid')
        self.fc2 = Linear(input_dim=64, output_dim=num_classes)
    # 网络的前向计算过程，定义输出每一层的结果，后续将结果写入VisualDL日志文件，实现每一层输出图片的可视化
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = fluid.layers.reshape(x5, [x5.shape[0], -1])
        x7 = self.fc1(x6)
        x8 = self.fc2(x7)
        # x8=F.sigmoid(x8)
        # conv=[x,x1,x2,x3,x4,x5,x6,x7,x8]
        return x8


# In[ ]:


class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):

        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y

# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        """

        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers,             "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]

        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)

        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                      weight_attr=paddle.ParamAttr(
                          initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y



# In[ ]:


class InceptionV4(paddle.nn.Layer):
    def __init__(self):
        super(InceptionV4,self).__init__()
        pass
    def name(self):
        """
        返回网络名字
        :return:
        """
        return 'InceptionV4'

    def forward(self, input,class_dim=1):
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
        # out=F.sigmoid(out)
        # if label is not None:
        #      acc = paddle.metric.accuracy(input=out, label=label)
        #      return out, acc
        # else:
        #      return out
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
    
  
 


# In[ ]:


# 查看数据形状
DATADIR = '/home/aistudio/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/PALM-Validation400'
CSVFILE = '/home/aistudio/data/data31387/labels.csv'
train_loader = data_loader(DATADIR, 
                           batch_size=10, mode='train')
data_reader = train_loader()
data = next(data_reader)
data[0].shape, data[1].shape


# In[ ]:





# In[ ]:


# 定义训练函数
def train(model,train_loader,epoch):
    print('start training ... ')
    # 是否使用GPU
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    model.train()
    for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            out = model(img)

            loss = F.binary_cross_entropy_with_logits(out, label)
            avg_loss = paddle.mean(loss)
            #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
            if batch_id % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy(),))
            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            opt.step()
            opt.clear_gradients()


# In[ ]:


# 定义对验证集的测试函数
def test(model,valid_loader):
    print('start testing ... ')
    model.eval()
    accuracies = []
    losses = []
    for batch_id, data in enumerate(valid_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        pred=F.sigmoid(logits)
        # 进行loss计算
        loss = F.binary_cross_entropy_with_logits(logits, label)

        pred2 = pred * (-1.0) + 1.0
        # 得到两个类别的预测概率，并沿第一个维度级联
        pred = paddle.concat([pred2, pred], axis=1)
        acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))
        accuracies.append(acc.numpy())
        losses.append(loss.numpy())
    print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
    # 保存模型参数
    paddle.save(model.state_dict(), 'palm.pdparams')
    # paddle.save(optimizer.state_dict(), 'palm.pdopt')
    return accuracies, losses


# In[ ]:


# 定义作图函数
def plot(accuracies,losses):
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plot_x=np.arange(len(accuracies))
    plt.plot(plot_x,accuracies)
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.title('validation accuracy')
    plt.subplot(2,1,2)
    plot_x=np.arange(len(losses))
    plt.plot(plot_x,losses)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('validation loss')
    plt.show()


# In[ ]:


def main(model):
    epoch_num = 5
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    accs,losses=[],[]
    for epoch in range(epoch_num):
        train(model,train_loader,epoch)
        acc,loss=test(model,valid_loader)
        accs.extend(acc)
        losses.extend(loss)
    plot(accs,losses)


# In[ ]:


model = ResNet()
main(model)


# In[ ]:





# In[ ]:





# In[ ]:


# evaluation(model,params_file_path='./model.pdparams')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
