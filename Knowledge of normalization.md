一、归一化--层归一化
   1、概念
         如果一个神经元的净输入分布在神经网络中是动态变化的，比如循环神经网络，那么无法应用批归一化操作。
层归一化和批归一化不同的是，层归一化是对一个中间层的所有神经元进行归一化。
        局部响应归一化（Local Response Normalization），简称LRN，是在深度学习中用于提高准确度，且一般会在激活、池化后进行的一种处理方法，通常应用于Alexnet中。
        归一化就是要把需要处理的数据经过处理后（通过某种算法）限制在你需要的一定范围内。首先归一化是为了后面数据处理的方便，其次是保证程序运行时收敛加快。归一化的具体作用是归纳统一样本的统计分布性。归一化在0-1之间是统计的概率分布，归一化在某个区间上是统计的坐标分布。归一化有同一、统一和合一的意思。
        归一化的目的简而言之，是使得没有可比性的数据变得具有可比性，同时又保持相比较的两个数据之间的相对关系，如大小关系；或是为了作图，原来很难在一张图上作出来，归一化后就可以很方便的给出图上的相对位置等。
   2.算法流程
         from keras_layer_normalization import LayerNormalization
   # 构建LN CNN网络
   model_ln = Sequential()
   model_ln.add(Conv2D(input_shape = (28,28,1), filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
   model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
   model_ln.add(Conv2D(input_shape=(14,14,6), filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
   model_ln.add(MaxPool2D(pool_size=(2,2), strides=2))
   model_ln.add(Flatten())
   model_ln.add(Dense(120, activation='tanh'))
   model_ln.add(LayerNormalization()) # 添加LN运算
   model_ln.add(Dense(84, activation='tanh'))
   model_ln.add(LayerNormalization())
   model_ln.add(Dense(10, activation='softmax'))
   from lstm_ln import LSTM_LN

   model_ln = Sequential()
   model_ln.add(Embedding(max_features,100))
   model_ln.add(LSTM_LN(128))
   model_ln.add(Dense(1, activation='sigmoid'))
   model_ln.summary()
   3、作用 
          批量归一化是对一个中间层的单个神经元进行归一化操作，因此要求小批量样本的数量不能太小，
否则难以计算单个神经元的统计信息。层归一化（Layer Normalization）是和批量归一化非常类似的方法。
和批量归一化不同的是，层归一化是对某一层的所有神经元进行归一化。
    4、应用场景--循环神经网络中的层归一化
          假设在时刻t，循环神经网络的隐藏层为 h，其归一化的更新为：
          z(t)=Uh(t-1)+Wx(t),h(t)=f(LN(r,p)z(t))
        x(t)为t时刻的输入， U,W 为网络参数人，r,p 代表缩放和平移的参数向量。
在标准循环神经网络中，循环神经网络的净输入一般会随着时间慢慢变大或变小，从而导致梯度爆炸或消失。
而层归一化的循环神经网络可以有效地缓解这种状况。对于 K个样本的一个小批量集合  ，层归一化是对每一列进行归一化，
而批量归一化是对每一行进行归一化。一般而言，批归一化是一种更好的选择，当小批量样本数量比较小时，可以选择层归一化。
二、可变形卷积
   1、DCN v1
    （1）概念：可变形卷积即卷积的位置是可变形的，并非在传统的N × N的网格上卷积，这样的好处是更准确地提取我们想要的特征。
    （2）实现：在传统的卷积操作上加入了一个偏移量（可以是小数），正是该偏移量才让卷积变形为不规则的卷积，所以其特征值要通过双线性插值方法来计算。
    （3）可变形池化： 变形池化的偏移量即是子区域的偏移。同理每一个子区域都有一个偏移，偏移量对应子区域有k×k个。
与可变形卷积不同的是，可变形池化的偏移量是通过全连接层得到的。
   2、DCN v2
     （1）引入：可变形卷积有可能引入了无用的上下文（区域）来干扰我们的特征提取，这显然会降低算法的表现。
     （2）改进：在DCN v1中只在conv 5中使用了三个可变形卷积，在DCN v2中把conv3到conv5都换成了可变形卷积，提高算法对几何形变的建模能力。
即，DCN v1中引入的offset是要寻找有效信息的区域位置，DCN v2中引入权重系数是要给找到的这个位置赋予权重，这两方面保证了有效信息的准确提取。
三、代码实践--眼疾识别原理
# -*- coding:utf-8 -*-
	
# ResNet模型代码
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
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
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

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

# 创建模型
model = ResNet()
# 定义优化器
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# 启动训练过程
train_pm(model, opt)



