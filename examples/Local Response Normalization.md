# 深度学习基础知识

## 局部响应归一化

### 概念：

**局部响应正则化（Local Response Normalization）用于对局部输入区域进行正则化，执行一种侧向抑制（lateral inhibition）** 

### 生物学基础：
侧抑制（lateral inhibitio），即指被激活的神经元抑制相邻的神经元。归一化（normaliazation）的目的就是“抑制”,LRN就是借鉴这种侧抑制来实现局部抑制，尤其是我们使用RELU的时候，这种“侧抑制”很有效 ，因而在alexnet里使用有较好的效果。
![](https://ai-studio-static-online.cdn.bcebos.com/206ce850358148858ef139aa02c094f8b0bf0c366ba3449696c1c7b707a7c467)

### 作用：
LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

### 公式及解释：
![](https://ai-studio-static-online.cdn.bcebos.com/8fe45c658b904f9a868d2da4bdc02f8a845afab65756404bb50c248ee5b34973)

首先，这个公式中的output表示卷积层（包括卷积操作和池化操作）后的输出结果，这个输出结果的结构是一个四维数组 [batch,height,width,channel]，

【batch:批次数(每一批为一张图片)

height:图片高度，

width：图片宽度，

channel：通道数可以理解成一批图片中的某一个图片经过卷积操作后输出的神经元个数(或是理解成处理后的图片深度)。】

[a,b,c,d]，作为在这个输出结构中的一个位置，即第a张图的第d个通道下的高度为b宽度为c的点。

size/2（n/2）：depth_radius  深度半径（涉及输出数据中通道的半径范围）

k：  bias   偏置
 
α,β： alpha,beta，表示函数中的两个控制参数

**其中size/2,k,α,β都是自定义的**

特别注意一下∑叠加的方向是沿着通道方向的，即每个点值的平方和是沿着input中的第3维channel方向的，也就是一个点同方向的前面size/2个通道（最小为第0个通道）和后size/2个通道（最大为第d-1个通道）的点的平方和(共size+1个点)。

i表示第i个核在位置（x,y）运用激活函数后的输出，n（size）是同一位置上临近的kernal map的数目，C是kernal的总数。

参数K,size,alpha，belta都是超参数，一般设置k=2,size=5,alpha=1Xe-4,beta=0.75。

解析图：
![](https://ai-studio-static-online.cdn.bcebos.com/c9e17739406a4b3b8eef26f752ea67fa5d3395404c0b4f94b81102e0478a739e)

参考文献：[LRN]：ImageNet Classification with Deep Convolutional Neural Networks

运行尝试：


```python
import paddle

x = paddle.rand(shape=(3, 3, 112, 112), dtype="float32")
y = paddle.nn.functional.local_response_norm(x, size=5)
#print(x)
print(y.shape)  # [3, 3, 112, 112]
```

    [3, 3, 112, 112]



```python

```
