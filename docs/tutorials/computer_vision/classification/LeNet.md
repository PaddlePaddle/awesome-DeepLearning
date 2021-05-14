# LeNet

LeNet是最早的卷积神经网络之一<sup>[1]</sup>。1998年，Yann LeCun第一次将LeNet卷积神经网络应用到图像分类上，在手写数字识别任务中取得了巨大成功。LeNet通过连续使用卷积和池化层的组合提取图像特征，其架构如 **图1** 所示，这里展示的是用于MNIST手写体数字识别任务中的LeNet-5模型：
<br></br>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/82e4124e2e6a4231bcde17e086bc86ba732d3e81dcd7415f86fb4ef050aa7772" width = "800"></center>
<center><br>图1：LeNet模型网络结构示意图</br></center>

<br></br>


* 第一模块：包含5×5的6通道卷积和2×2的池化。卷积提取图像中包含的特征模式（激活函数使用Sigmoid），图像尺寸从28减小到24。经过池化层可以降低输出特征图对空间位置的敏感性，图像尺寸减到12。

* 第二模块：和第一模块尺寸相同，通道数由6增加为16。卷积操作使图像尺寸减小到8，经过池化后变成4。

* 第三模块：包含4×4的120通道卷积。卷积之后的图像尺寸减小到1，但是通道数增加为120。将经过第3次卷积提取到的特征图输入到全连接层。第一个全连接层的输出神经元的个数是64，第二个全连接层的输出神经元个数是分类标签的类别数，对于手写数字识别的类别数是10。然后使用Softmax激活函数即可计算出每个类别的预测概率。

------

**【提示】：**

卷积层的输出特征图如何当作全连接层的输入使用呢？

卷积层的输出数据格式是$[N, C, H, W]$，在输入全连接层的时候，会自动将数据拉平，

也就是对每个样本，自动将其转化为长度为$K$的向量，

其中$K = C \times H \times W$，一个mini-batch的数据维度变成了$N\times K$的二维向量。

------

## LeNet在手写数字识别上的应用

LeNet网络的实现代码如下：


```python
# 导入需要的包
import paddle
import numpy as np
from paddle.nn import Conv2D, MaxPool2D, Linear

## 组网
import paddle.nn.functional as F

# 定义 LeNet 网络结构
class LeNet(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x
```

## 参考文献

[1] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learn- ing applied to document recognition. Proc. of the IEEE, 86(11):2278–2324, 1998 