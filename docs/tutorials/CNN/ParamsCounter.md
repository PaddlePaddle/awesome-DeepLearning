# CNN中模型的参数量与FLOPs计算
一个卷积神经网络的基本构成一般有卷积层、归一化层、激活层和线性层。这里我们就通过逐步计算这些层来计算一个CNN模型所需要的参数量和FLOPs吧. 另外，FLOPs的全程为floating point operations的缩写（小写s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
## 1. 卷积层
卷积层，最常用的是2D卷积，因此我们以飞桨中的Conv2D来表示。

### 1.1 卷积层参数量计算
Conv2D的参数量计算较为简单，先看下列的代码，如果定义一个Conv2D，卷积层中的参数会随机初始化，如果打印其shape，就可以知道一个Conv2D里大致包含的参数量了，Conv2D的参数包括两部分，一个是用于卷积的weight，一个是用于调节网络拟合输入特征的bias。如下
```python
import paddle
import numpy as np
cv2d   = paddle.nn.Conv2D(in_channels=2, out_channels=4, kernel_size=(3, 3), stride=1, padding=(1, 1))
params = cv2d.parameters()
print("shape of weight: ", np.array(params[0]).shape)
print("shape of bias: ", np.array(params[1]).shape)
shape of weight:  (4, 2, 3, 3)
shape of bias:  (4,)
```
这里解释一下上面的代码，我们先定义了一个卷积层cv2d，然后输出了这个卷积层的参数的形状，参数包含两部分，分别是weight和bias，这两部分相加才是整个卷积的参数量。因此，可以看到，我们定义的cv2d的参数量为：$4*2*3*3+4 = 76$, 4对应的是输出的通道数，2对应的是输入的通道数，两个3是卷积核的尺寸，最后的4就是bias的数量了， 值得注意的是， bias是数量与输出的通道数保持一致。因此，我们可以得出，一个卷积层的参数量的公式，如下：
$$ Param_{conv2d} = C_{in} * C_{out} * K_h * K_w + C_{out} $$
其中，$C_{in}$ 表示输入的通道数，$C_{out}$ 表示输出的通道数, $K_h$, $K_w$ 表示卷积核的大小。当然了，有些卷积会将bias设置为False，那么我们不加最后的$C_{out}$即可。

### 1.2 卷积层FLOPs计算
参数量会计算了，那么FLOPs其实也是很简单的，就一个公式：
$$FLOP_{conv2d} = Param_{conv2d} * M_{outh} * M_{outw}$$
这里，$M_{outh}$，$M_{outw}$ 为输出的特征图的高和宽，而不是输入的，这里需要注意一下。

### 1.3 卷积层参数计算示例
Paddle有提供计算FLOPs和参数量的API，[paddle.flops](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/flops_cn.html#flops), 这里我们用我们的方法和这个API的方法来测一下，看看一不一致吧。代码如下：
```python
import paddle
from paddle import nn
from paddle.nn import functional as F


class TestNet(nn.Layer):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv2d = nn.Conv2D(in_channels=2, out_channels=4, kernel_size=(3, 3), stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d(x)
        return x
        

if __name__ == "__main__":
    net = TestNet()
    paddle.flops(net, input_size=[1, 2, 320, 320])
    
    
Total GFlops: 0.00778     Total Params: 76.00
```
API得出的参数量为76，GFLOPs为0.00778，这里的GFLOPs就是FLOPs的10$^9$倍，我们的参数量求得的也是76，那么FLOPs呢？我们来算一下，输入的尺寸为320 * 320， 卷积核为3 * 3， 且padding为1，那么图片输入的大小和输出的大小一致，即输出也是320 * 320， 那么根据我们的公式可得: $76 * 320 * 320 = 7782400$, 与API的一致！因此大家计算卷积层的参数和FLOPs的时候就可以用上面的公式。
## 2. 归一化层
最常用的归一化层为BatchNorm2D啦，我们这里就用BatchNorm2D来做例子介绍。在算参数量和FLOPs，先看看BatchNorm的算法流程吧！

输入为：Values of $x$ over a mini-batch:$B={x_1,...,m}$,

$\quad\quad\quad$Params to be learned: $\beta$, $\gamma$

输出为：{$y_i$=BN$_{\gamma}$,$\beta(x_i)$}

流程如下：

$\quad\quad\quad$$\mu_B\gets$ $\frac{1}{m}\sum_{1}^mx_i$

$\quad\quad\quad\sigma_{B}^2\gets\frac{1}{m}\sum_{1}^m(x_i-\mu_B)^2$

$\quad\quad\quad\hat{x}_i\gets\frac{x_i-\mu_B}{\sqrt{\sigma_{B}^2+\epsilon}}$

$\quad\quad\quad y_i\gets\gamma\hat{x}_i+\beta\equiv BN_{\gamma}$,$\beta(x_i)$

在这个公式中，$B$ 为一个Batch的数据，$\beta$ 和 $\gamma$ 为可学习的参数，$\mu$ 和 $\sigma^2$ 为均值和方差，由输入的数据的值求得。该算法先求出整体数据集的均值和方差，然后根据第三行的更新公式求出新的x，最后根据可学习的$\beta$ 和 $\gamma$调整数据。第三行中的 $\epsilon$ 在飞桨中默认为 1e-5， 用于处理除法中的极端情况。

### 2.1 归一化层参数量计算
由于归一化层较为简单，这里直接写出公式：
$$Param_{bn2d} = 4 * C_{out} $$
其中4表示四个参数值，每个特征图对应一组四个元素的参数组合；

beta_initializer $\beta$ 权重的初始值设定项。

gamma_initializer $\gamma$ 伽马权重的初始值设定项。

moving_mean_initializer $\mu$ 移动均值的初始值设定项。

moving_variance_initializer $\sigma^2$ 移动方差的初始值设定项。
### 2.2 归一化层FLOPs计算
因为只有两个可以学习的权重，$\beta$ 和 $\gamma$，所以FLOPs只需要2乘以输出通道数和输入的尺寸即可。
归一化的FLOPs计算公式则为:
$$ FLOP_{bn2d} = 2 * C_{out} * M_{outh} * M_{outw} $$
与1.3相似，欢迎大家使用上面的代码进行验证。

## 3. 线性层
线性层也是常用的分类层了，我们以飞桨的Linear为例来介绍。
### 3.1 线性层参数量计算
其实线性层是比较简单的，它就是相当于卷积核为1的卷积层，线性层的每一个参数与对应的数据进行矩阵相乘，再加上偏置项bias，线性层没有类似于卷积层的“卷”的操作的，所以计算公式如下：
$$Param_{linear} = C_{in} * C_{out}  + C_{out} $$。我们这里打印一下线性层参数的形状看看。
```python
import paddle
import numpy as np
linear = paddle.nn.Linear(in_features=2, out_features=4)
params = linear.parameters()
print("shape of weight: ", np.array(params[0]).shape)
print("shape of bias: ", np.array(params[1]).shape)

shape of weight:  (2, 4)
shape of bias:  (4,)
```
可以看到，线性层相较于卷积层还是简单的，这里我们直接计算这个定义的线性层的参数量为 $2 * 4 + 4 = 12$。具体对不对，我们在下面的实例演示中检查。
### 3.2 线性层FLOPs计算
与卷积层不同的是，线性层没有”卷“的过程，所以线性层的FLOPs计算公式为:
$$ FLOP_{linear} = C_{in} * C_{out}$$

## 4. 实例演示
这里我们就以LeNet为例子，计算出LeNet的所有参数量和计算量。LeNet的结构如下。输入的图片大小为28 * 28
```python
LeNet(
  (features): Sequential(
    (0): Conv2D(1, 6, kernel_size=[3, 3], padding=1, data_format=NCHW)
    (1): ReLU()
    (2): MaxPool2D(kernel_size=2, stride=2, padding=0)
    (3): Conv2D(6, 16, kernel_size=[5, 5], data_format=NCHW)
    (4): ReLU()
    (5): MaxPool2D(kernel_size=2, stride=2, padding=0)
  )
  (fc): Sequential(
    (0): Linear(in_features=400, out_features=120, dtype=float32)
    (1): Linear(in_features=120, out_features=84, dtype=float32)
    (2): Linear(in_features=84, out_features=10, dtype=float32)
  )
)
我们先来手动算一下参数量和FLOPs。

```

features[0] 参数量：$ 6 * 1 * 3 * 3  + 6 = 60$, FLOPs : $ 60 * 28 * 28 = 47040$

features[1] 参数量和FLOPs均为0

features[2] 参数量和FLOPs均为0， 输出尺寸变为14 * 14

features[3] 参数量：$ 16 * 6 * 5 * 5  + 16 = 2416$, FLOPs : $ 2416 * 10 * 10 = 241600$, 需要注意的是，这个卷积没有padding，所以输出特征图大小变为 10 * 10 

features[4] 参数量和FLOPs均为0

features[5] 参数量和FLOPs均为0，输出尺寸变为5 * 5， 然后整个被拉伸为[1, 400]的尺寸，其中400为5 * 5 * 16。

fc[0] 参数量：$ 400 * 120  + 120 = 48120$, FLOPs : $ 400 * 120  = 48000$ （输出尺寸变为[1, 120]）

fc[1] 参数量：$ 120 * 84  + 84 = 10164$, FLOPs : $ 120 * 84 = 10080$ （输出尺寸变为[1, 84]）

fc[2] 参数量：$ 84 * 10  + 10 = 850$, FLOPs : $ 84 * 10 = 840 $ （输出尺寸变为[1, 10]）。

总参数量为： $60 + 2416 + 48120 + 10164 + 850 = 61610$

总FLOPs为：$47040 + 241600 + 48000 + 10080 + 840 = 347560$

下面我们用代码验证以下：
```python
from paddle.vision.models import LeNet
net = LeNet()
print(net)
paddle.flops(net, input_size=[1, 1, 28, 28], print_detail=True)

+--------------+-----------------+-----------------+--------+--------+
|  Layer Name  |   Input Shape   |   Output Shape  | Params | Flops  |
+--------------+-----------------+-----------------+--------+--------+
|   conv2d_0   |  [1, 1, 28, 28] |  [1, 6, 28, 28] |   60   | 47040  |
|   re_lu_0    |  [1, 6, 28, 28] |  [1, 6, 28, 28] |   0    |   0    |
| max_pool2d_0 |  [1, 6, 28, 28] |  [1, 6, 14, 14] |   0    |   0    |
|   conv2d_1   |  [1, 6, 14, 14] | [1, 16, 10, 10] |  2416  | 241600 |
|   re_lu_1    | [1, 16, 10, 10] | [1, 16, 10, 10] |   0    |   0    |
| max_pool2d_1 | [1, 16, 10, 10] |  [1, 16, 5, 5]  |   0    |   0    |
|   linear_0   |     [1, 400]    |     [1, 120]    | 48120  | 48000  |
|   linear_1   |     [1, 120]    |     [1, 84]     | 10164  | 10080  |
|   linear_2   |     [1, 84]     |     [1, 10]     |  850   |  840   |
+--------------+-----------------+-----------------+--------+--------+
Total GFlops: 0.00034756     Total Params: 61610.00

```
可以看到，与我们的计算是一致的，大家可以自己把VGG-16的模型算一下参数量FLOPs，相较于LeNet， VGG-16只是模型深了点，并没有其余额外的结构。
