#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# # VisualDL2.0应用案例--眼疾识别训练可视化
# 
# 本项目将基于眼疾分类数据集[iChallenge-PM](https://ai.baidu.com/broad/introduction)，介绍如何运用飞桨可视化分析工具--VisualDL对模型训练过程进行可视化分析。
# 
# VisualDL是深度学习模型可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。帮助用户清晰直观地理解模型训练过程及模型结构，进而实现高效的模型调优。VisualDL的具体介绍请参考：[GitHub](https://github.com/PaddlePaddle/VisualDL)、[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README.md)。
# 
# iChallenge-PM是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。下面我们将详细介绍如何使用VisualDL进行：
# 
# - 创建日志文件
# - 实时训练参数可视化
# - 展示多组实验训练参数对比
# - 训练数据中间状态可视化

# ### 数据集准备
# /home/aistudio/data/data19065 目录包括如下三个文件，解压缩后存放在/home/aistudio/work/palm目录下。
# - training.zip：包含训练中的图片和标签
# - validation.zip：包含验证集的图片
# - valid_gt.zip：包含验证集的标签
# - valid_gt.zip文件解压缩之后，需要将/home/aistudio/work/palm/PALM-Validation-GT/目录下的PM_Label_and_Fovea_Location.xlsx文件转存成csv格式，本节代码示例中已经提前转成文件labels.csv。
# 
# 

# In[ ]:


# 初次运行时将注释取消，以便解压文件
# 如果已经解压过了，则不需要运行此段代码，否则文件已经存在解压会报错
get_ipython().system('unzip -oq -d /home/aistudio/work/palm /home/aistudio/data/data92378/valid_gt.zip')
get_ipython().system('unzip -oq -d /home/aistudio/work/palm /home/aistudio/data/data92378/validation.zip')
get_ipython().system('unzip -oq -d /home/aistudio/work/palm /home/aistudio/data/data92378/training.zip')



# In[ ]:


get_ipython().run_line_magic('cd', '/home/aistudio/work/palm/PALM-Training400/')
get_ipython().system('unzip -o -q PALM-Training400.zip')


# ### 使用VisualDL查看数据集
# 
# iChallenge-PM中既有病理性近视患者的眼底图片，也有非病理性近视患者的图片，命名规则如下：
# 
# - 病理性近视（PM）：文件名以P开头
# 
# - 非病理性近视（non-PM）：
# 
#   * 高度近似（high myopia）：文件名以H开头
#   
#   * 正常眼睛（normal）：文件名以N开头
# 
# 我们将病理性患者的图片作为正样本，标签为1； 非病理性患者的图片作为负样本，标签为0。从数据集中选取两张图片，通过LeNet提取特征，构建分类器，对正负样本进行分类，并将图片在VisualDL中显示出来。代码如下所示：
# 

# In[ ]:


import numpy as np
from PIL import Image
from visualdl import LogWriter
import os

#确保路径为'/home/aistudio'
os.chdir('/home/aistudio')

#创建 LogWriter 对象，将图像数据存放在 `./log/train`路径下
from visualdl import LogWriter
log_writer = LogWriter("./log/train")

#导入所需展示的图片
img1 = Image.open('work/palm/PALM-Training400/PALM-Training400/N0012.jpg')
img2 = Image.open('work/palm/PALM-Training400/PALM-Training400/P0095.jpg')

#将图片转化成array格式
img_n1=np.asarray(img1)
img_n2=np.asarray(img2)

#将图片数据打点至日志文件
log_writer.add_image(tag='图像样本/正样本',img=img_n2, step=5)
log_writer.add_image(tag='图像样本/负样本',img=img_n1, step=5)


# ### 定义数据读取器
# 
# 使用OpenCV从磁盘读入图片，将每张图缩放到$224\times224$大小，并且将像素值调整到$[-1, 1]$之间，代码如下所示：

# In[ ]:


import cv2
import random
import numpy as np
import os

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

# 定义训练集数据读取器
def data_loader(datadir, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    filenames = os.listdir(datadir)  # filenames是图片的名字的列表
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)  # 图片的路径
            img = cv2.imread(filepath)  # 读取图片
            img = transform_img(img)  # 图片格式转换，每张图片是三维，转换后[3,224,224]
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
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)  # 标签只有一列，-1是自适应维度的意思
                yield imgs_array, labels_array  # yield的函数是一个迭代器，
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

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


# 查看数据形状
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
train_loader = data_loader(DATADIR, 
                           batch_size=10, mode='train')
data_reader = train_loader()  # 生成器
data = next(data_reader)  # 生成器要取得值要用next函数
data[0].shape, data[1].shape


# ## 使用ResNet网络进行眼疾分类
# 
# - ResNet-50的具体实现如下代码所示：

# # 1.1、 RestNet网络结构
# 
# ResNet在2015年被提出，在ImageNet比赛classification任务上获得第一名，因为它“简单与实用”并存，之后很多方法都建立在ResNet50或者ResNet101的基础上完成的，检测，分割，识别等领域里得到广泛的应用。它使用了一种连接方式叫做“shortcut connection”，顾名思义，shortcut就是“抄近道”的意思，下面是这个resnet的网络结构：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/3f84dd1c19354469bd031e11a6e15ea75ed32c78632e4a59b666edbaba70ee75)
# 
# 它对每层的输入做一个reference（X）, 学习形成残差函数， 而不是学习一些没有reference（X）的函数。这种残差函数更容易优化，能使网络层数大大加深。在上图的残差块中它有二层，如下表达式，
# 其中σ代表非线性函数ReLU。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/a03b29f9c1044b10a259a74259c951e7539dbd82be254a24b6b769fef853a9b7)
# 
# 然后通过一个shortcut，和第2个ReLU，获得输出y。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/5f4e321e2a3f4e22a54310a2f941e0adbb1434b1561541c79f42aa4d5f90a85d)
# 
# 当需要对输入和输出维数进行变化时（如改变通道数目），可以在shortcut时对x做一个线性变换Ws，如下式。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/68df3227441d48d0b7d84a988944d627619523ec701c4b0cb4c2ab61cd1ac149)
# 
# 然而实验证明x已经足够了，不需要再搞个维度变换，除非需求是某个特定维度的输出，如是将通道数翻倍，如下图所示：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/98d9929829914ab2a9a6b096737787462c13a2e76b1444689df94d242b089de8)
# 
# 由上图，我们可以清楚的看到“实线”和“虚线”两种连接方式， 实线的Connection部分 (第一个粉色矩形和第三个粉色矩形) 都是执行3x3x64的卷积，他们的channel个数一致，所以采用计算方式：
# Y = F(x) + x，虚线的Connection部分 (第一个绿色矩形和第三个绿色矩形) 分别是3x3x64和3x3x128的卷积操作，他们的channel个数不同(64和128)，所以采用计算方式： y=F(x)+Wx 。其中W是卷积操作，用来调整x的channel维度。
# 
# # 1.2、残差块的两种结构
# 
# 这是文章里面的图，我们可以看到一个“弯弯的弧线“这个就是所谓的”shortcut connection“，也是文中提到identity mapping，这张图也诠释了ResNet的真谛，当然大家可以放心，真正在使用的ResNet模块并不是这么单一，文章中就提出了两种方式：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/814e1728b7954301801671bb3129742683cdf02e4e624fc3af2d5da59ecd3181)
# 
# 这两种结构分别针对ResNet34（左图）和ResNet50/101/152（右图），一般称整个结构为一个“building block” 。其中右图又称为“bottleneck design”，目的就是为了降低参数的数目，实际中，考虑计算的成本，对残差块做了计算优化，即将两个3x3的卷积层替换为1x1 + 3x3 + 1x1，如右图所示。新结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，而不使用bottleneck的话就是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍。
# 对于常规ResNet，可以用于34层或者更少的网络中，对于Bottleneck Design的ResNet通常用于更深的如101这样的网络中，目的是减少计算和参数量。
# 
# # 1.3、 ResNet50模型
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/b54689af4743490cb7c86a1589089c19954e6147c1454023b4b5a9ea035d6528)
# 
# 

# In[ ]:


# -*- coding:utf-8 -*-

# ResNet模型代码
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(fluid.dygraph.Layer):
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
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(fluid.dygraph.Layer):
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

        y = fluid.layers.elementwise_add(x=short, y=conv2)
        layer_helper = LayerHelper(self.full_name(), act='relu')
        return layer_helper.append_activation(y)

# 定义ResNet模型
class ResNet(fluid.dygraph.Layer):
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
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
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
        self.pool2d_max = Pool2D(
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

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
        self.pool2d_avg = Pool2D(pool_size=7, pool_type='avg', global_pooling=True)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目
        self.out = Linear(input_dim=2048, output_dim=class_dim,
                      param_attr=fluid.param_attr.ParamAttr(
                          initializer=fluid.initializer.Uniform(-stdv, stdv)))

        
    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y1 = self.pool2d_avg(y)
        y2 = fluid.layers.reshape(y1, [y1.shape[0], -1])
        y3 = self.out(y2)
        conv=[inputs,y1,y2,y3]
        return y3,conv


# ### 训练模型并使用VisualDL可视化训练参数及数据样本
# - 创建ResNet日志文件，以便对比其他模型训练参数，代码如下：
# 
# log_writer = LogWriter("./log/resenet")
# 
# - 训练过程中插入作图语句，展示accuracy和loss的变化趋势：
# 
# log_writer.add_scalar(tag='acc', step=iter, value=acc.numpy())
# 
# log_writer.add_scalar(tag='loss', step=iter, value=avg_loss.numpy())
# 
# - 设计网络向前计算过程时，将每一层的输出储存于名为'conv'的list中，方便后续写入日志文件
# 
# - 训练过程中插入作图语句，展示输入图片在每一层网络的输出
# 
# log_writer.add_image(tag='input_resnet/pool2d_avg', img=conv[0].numpy(), step=batch_id)
# 
# ***注意使用相同tag才能实现多组模型实验对比**
# 
# #### 完整训练及可视化代码如下：

# In[33]:


import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np


#定义文件路径
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/work/palm/PALM-Validation400'
CSVFILE = '/home/aistudio/work/palm/PALM-Validation-GT/labels.csv'

#创建储存resnet结果的日志文件夹
log_writer = LogWriter("./log/resnet1.log")

# 定义训练过程
def train(model):
    with fluid.dygraph.guard():
        print('start training ... ')

        # 开始训练
        model.train()
        epoch_num = 3  #训练周期
        iter=0
        # 定义优化器
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
        # 定义数据读取器，训练数据读取器和验证数据读取器
        train_loader = data_loader(DATADIR, batch_size=10, mode='train')  # 训练集读取器
        valid_loader = valid_data_loader(DATADIR2, CSVFILE)  # 验证集读取器
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)  # 预测值
                # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)  
                pred2 = pred * (-1.0) + 1.0
                pred = fluid.layers.concat([pred2, pred], axis=1)
                #计算accuracy
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                # 进行loss计算
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
                avg_loss = fluid.layers.mean(loss)
                #训练过程中插入作图语句，当每10个batch训练完成后，将当前损失、准确率作为一个新增的数据点储存到记录器中。
                if (batch_id + 1) % 10 == 0:
                    log_writer.add_scalar(tag='train/acc', step=iter, value=acc.numpy())  # 训练集准确率
                    log_writer.add_scalar(tag='train/loss', step=iter, value=avg_loss.numpy())  # 训练集损失
                    iter+=10
                    print("epoch: {}, batch_id: {}, loss is: {}".format(epoch+1, batch_id + 1, avg_loss.numpy()))
                # 反向传播，更新权重，清除梯度
                avg_loss.backward()
                opt.minimize(avg_loss)
                model.clear_gradients()
            
            # 当所有训练集样本训练完之后，在调用验证集进行验证，训练集400个，验证集400个
            model.eval()
            accuracies = []  # 储存验证集历史准确率
            losses = []  # 储存验证集历史损失
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                # 运行模型前向计算，得到预测值
                logits,conv = model(img)
                # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
                # 计算sigmoid后的预测概率，进行loss计算
                pred = fluid.layers.sigmoid(logits)
                loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)  # 交叉熵损失
                avg_loss = fluid.layers.mean(loss)  # 由于一个mini-batch是10个样本，所以要取平均
                # 计算预测概率小于0.5的类别
                pred2 = pred * (-1.0) + 1.0
                # 得到两个类别的预测概率，并沿第一个维度级联
                pred = fluid.layers.concat([pred2, pred], axis=1)
                acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
                accuracies.append(acc.numpy())
                losses.append(loss.numpy())
            print("[validation] accuracy:{} loss:{}".format(np.mean(accuracies), np.mean(losses)))
            model.train()

        # save params of model
        fluid.save_dygraph(model.state_dict(), 'ResNet50')
        # save optimizer state
        fluid.save_dygraph(opt.state_dict(), 'resnet50')

with fluid.dygraph.guard():
    model = ResNet()

train(model)


# # 实验结果
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/32df2904896f4307b05b6cca7c44031038f8bc85f1a4406a990fe6ee19960279)
# 
# 可以看到，最后的准确率大概是94%,损失函数降到了0.18

# 
