# ResNet

ResNet是2015年ImageNet比赛的冠军，将识别错误率降低到了3.6%，这个结果甚至超出了正常人眼识别的精度。

通过前面几个经典模型学习，我们可以发现随着深度学习的不断发展，模型的层数越来越多，网络结构也越来越复杂。那么是否加深网络结构，就一定会得到更好的效果呢？从理论上来说，假设新增加的层都是恒等映射，只要原有的层学出跟原模型一样的参数，那么深模型结构就能达到原模型结构的效果。换句话说，原模型的解只是新模型的解的子空间，在新模型解的空间里应该能找到比原模型解对应的子空间更好的结果。但是实践表明，增加网络的层数之后，训练误差往往不降反升。

Kaiming He等人提出了残差网络ResNet来解决上述问题，其基本思想如 **图6**所示。
* 图6(a)：表示增加网络的时候，将$x$映射成$y=F(x)$输出。
* 图6(b)：对图6(a)作了改进，输出$y=F(x) + x$。这时不是直接学习输出特征$y$的表示，而是学习$y-x$。
  - 如果想学习出原模型的表示，只需将$F(x)$的参数全部设置为0，则$y=x$是恒等映射。
  - $F(x) = y - x$也叫做残差项，如果$x\rightarrow y$的映射接近恒等映射，图6(b)中通过学习残差项也比图6(a)学习完整映射形式更加容易。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e10f22f054704daabf4261ab46719629a36749631db74eb0a368499de3e5d3d6" width = "500"></center>
<center><br>图6：残差块设计思想</br></center>
<br></br>


图6(b)的结构是残差网络的基础，这种结构也叫做残差块（Residual block）。输入$x$通过跨层连接，能更快的向前传播数据，或者向后传播梯度。通俗的比喻，在火热的电视节目《王牌对王牌》上有一个“传声筒”的游戏，排在队首的嘉宾把看到的影视片段表演给后面一个嘉宾看，经过四五个嘉宾后，最后一个嘉宾如果能表演出更多原剧的内容，就能取得高分。我们常常会发现刚开始的嘉宾往往表演出最多的信息（类似于Loss），而随着表演的传递，有效的表演信息越来越少（类似于梯度弥散）。如果每个嘉宾都能看到原始的影视片段，那么相信传声筒的效果会好很多。类似的，由于ResNet每层都存在直连的旁路，相当于每一层都和最终的损失有“直接对话”的机会，自然可以更好的解决梯度弥散的问题。残差块的具体设计方案如 **图**7 所示，这种设计方案也常称作瓶颈结构（BottleNeck）。1\*1的卷积核可以非常方便的调整中间层的通道数，在进入3\*3的卷积层之前减少通道数（256->64），经过该卷积层后再恢复通道数(64->256)，可以显著减少网络的参数量。这个结构（256->64->256）像一个中间细，两头粗的瓶颈，所以被称为“BottleNeck”。
<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/322b26358d43401ba81546dd134a310cfb11ecafb3314aab88b5885ff642870b" width = "500"></center>
<center><br>图7：残差块结构示意图</br></center>
<br></br>


下图表示出了ResNet-50的结构，一共包含49层卷积和1层全连接，所以被称为ResNet-50。

<br></br>
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8f42b3b5b7b34e45847a9c61580f1f8239a80ca6fa67448e8baeeb0209a2d556" width = "1000"></center>
<center><br>图8：ResNet-50模型网络结构示意图</br></center>
<br></br>

ResNet-50的具体实现如下代码所示：


```python
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

```


```python
# 创建模型
model = ResNet()
# 定义优化器
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# 启动训练过程
train_pm(model, opt)
```

    start training ... 


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:636: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")


    epoch: 0, batch_id: 0, loss is: [0.70181155]
    epoch: 0, batch_id: 10, loss is: [0.6785618]
    epoch: 0, batch_id: 20, loss is: [0.6985228]
    epoch: 0, batch_id: 30, loss is: [0.56165934]
    [validation] accuracy/loss: 0.762499988079071/0.537024199962616
    epoch: 1, batch_id: 0, loss is: [0.58991754]
    epoch: 1, batch_id: 10, loss is: [0.37351575]
    epoch: 1, batch_id: 20, loss is: [0.35140988]
    epoch: 1, batch_id: 30, loss is: [0.37881708]
    [validation] accuracy/loss: 0.6325000524520874/0.7953802347183228
    epoch: 2, batch_id: 0, loss is: [0.46120423]
    epoch: 2, batch_id: 10, loss is: [0.21911184]
    epoch: 2, batch_id: 20, loss is: [0.41942623]
    epoch: 2, batch_id: 30, loss is: [0.08519388]
    [validation] accuracy/loss: 0.9300000071525574/0.20144781470298767
    epoch: 3, batch_id: 0, loss is: [0.18799873]
    epoch: 3, batch_id: 10, loss is: [0.22485955]
    epoch: 3, batch_id: 20, loss is: [0.68815064]
    epoch: 3, batch_id: 30, loss is: [0.44796553]
    [validation] accuracy/loss: 0.9275000691413879/0.19075711071491241
    epoch: 4, batch_id: 0, loss is: [0.2415458]
    epoch: 4, batch_id: 10, loss is: [0.1657328]
    epoch: 4, batch_id: 20, loss is: [0.96144015]
    epoch: 4, batch_id: 30, loss is: [0.07468574]
    [validation] accuracy/loss: 0.9300000071525574/0.2081790417432785


通过运行结果可以发现，使用ResNet在眼疾筛查数据集iChallenge-PM上，loss能有效的下降，经过5个epoch的训练，在验证集上的准确率可以达到95%左右。
