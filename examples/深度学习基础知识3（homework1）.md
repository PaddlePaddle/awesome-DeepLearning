深度学习基础知识3:

1归一化方法详解:

局部响应归一化

概念:

局部响应归一化层简称LRN，是在深度学习训练中提高准确度的技术方法。一般是在激活、池化后进行的一种处理方法。

算法流程:

The 4-D input tensor is treated as a 3-D array of 1-D vectors (along the last dimension), and each vector is normalized independently. Within a given vector, each component is divided by the weighted, squared sum of inputs within depth_radius. In detail, 
sqr_sum[a, b, c, d] = 
sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2) 
output = input / (bias + alpha * sqr_sum) ** beta

局部响应归一化原理是仿造生物学上活跃的神经元对相邻神经元的抑制现象（侧抑制），然后根据论文有公式如下 
![20170713145228303](C:\Users\86133\Desktop\images\20170713145228303.png)

这个公式中的a表示卷积层（包括卷积操作和池化操作）后的输出结果，这个输出结果的结构是一个四维数组[batch,height,width,channel]，这里可以简单解释一下，batch就是批次数(每一批为一张图片)，height就是图片高度，width就是图片宽度，channel就是通道数可以解成一批图片中的某一个图片经过卷积操作后输出的神经元个数(或是理解成处理后的图片深度)。ai(x,y)表示在这个输出结构中的一个位置[a,b,c,d]，可以理解成在某一张图中的某一个通道下的某个高度和某个宽度位置的点，即第a张图的第d个通道下的高度为b宽度为c的点。论文公式中的N表示通道数(channel)。a,n/2,k,α,β分别表示函数中的input,depth_radius,bias,alpha,beta，其中n/2,k,α,β都是自定义的，特别注意一下∑叠加的方向是沿着通道方向的，即每个点值的平方和是沿着a中的第3维channel方向的，也就是一个点同方向的前面n/2个通道（最小为第0个通道）和后n/2个通道（最大为第d-1个通道）的点的平方和(共n+1个点)。而函数的英文注解中也说明了把input当成是d个3维的矩阵，说白了就是把input的通道数当作3维矩阵的个数，叠加的方向也是在通道方向。 

```python
import tensorflow as tf
import numpy as np
x = np.array([i for i in range(1,33)]).reshape([2,2,2,4])

y = tf.nn.lrn(input=x,depth_radius=2,bias=0,alpha=1,beta=1)
with tf.Session() as sess:
    print(x)
    print('#############')
    print(y.eval())
```

作用:

首先要引入一个神经生物学的概念：侧抑制（lateral inhibitio），即指被激活的神经元抑制相邻的神经元。归一化（normaliazation）的目的就是“抑制”,LRN就是借鉴这种侧抑制来实现局部抑制，尤其是我们使用RELU的时候，这种“侧抑制”很有效 ，因而在alexnet里使用有较好的效果。

对局部神经元的活动创建竞争机制,使得其中响应比较大的值变得相对更大,并抑制其他反馈较小的神经元,增强了模型的泛化能力。

应用场景:

LRN首次是在2012的AlexNet当中使用，其中的意图是对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

2可变形卷积方法详解:

v1:

背景：

在计算机视觉领域，同一物体在不同场景，角度中未知的几何变换是检测/识别的一大挑战，通常来说我们有两种做法:

(1)通过充足的数据增强，扩充足够多的样本去增强模型适应尺度变换的能力。

(2)设置一些针对几何变换不变的特征或者算法，比如SIFT和sliding windows。

两种方法都有缺陷，第一种方法*因为样本的局限性显然模型的泛化能力比较低，无法泛化到一般场景中*，第二种方法则*因为手工设计的不变特征和算法对于过于复杂的变换是很难的而无法设计*。所以作者提出了Deformable Conv（可变形卷积）和 Deformable Pooling（可变形池化）来解决这个问题。

可变形卷积：

可变形卷积顾名思义就是卷积的位置是可变形的，并非在传统的N × N的网格上做卷积，这样的好处就是更准确地提取到我们想要的特征（传统的卷积仅仅只能提取到矩形框的特征），通过一张图我们可以更直观地了解：

![v2-4858fb10ed4ab3bf3ff6744920fd8473_720w](C:\Users\86133\Desktop\images\v2-4858fb10ed4ab3bf3ff6744920fd8473_720w.jpg)

在上面这张图里面，左边传统的卷积显然没有提取到完整绵羊的特征，而右边的可变形卷积则提取到了完整的不规则绵羊的特征。

那可变卷积实际上是怎么做的呢？*其实就是在每一个卷积采样点加上了一个偏移量*，如下图所示：

![img](C:\Users\86133\Desktop\images\v2-6509faf5c740ea9005e8fea2d979edba_720w.jpg)

(a) 所示的正常卷积规律的采样 9 个点（绿点），(b)(c)(d) 为可变形卷积，在正常的采样坐标上加上一个位移量（蓝色箭头），其中 (c)(d) 作为 (b) 的特殊情况，展示了可变形卷积可以作为尺度变换，比例变换和旋转变换等特殊情况。

我们先看普通的卷积，以3x3卷积为例对于每个输出y(p0)，都要从x上采样9个位置，这9个位置都在中心位置x(p0)向四周扩散，(-1,-1)代表x(p0)的左上角，(1,1)代表x(p0)的右下角。

![v2-313a0261ab05543452368f3f415a3b50_720w](C:\Users\86133\Desktop\images\v2-313a0261ab05543452368f3f415a3b50_720w.png)

所以传统的卷积输出就是（其中![[公式]](https://www.zhihu.com/equation?tex=P_n)就是网格中的n个点）：

![v2-00f39b1b57c659d1b3e8b2e8684d0c2f_720w](C:\Users\86133\Desktop\images\v2-00f39b1b57c659d1b3e8b2e8684d0c2f_720w.jpg)

正如我们上面阐述的可变形卷积，他就是在传统的卷积操作上加入了一个偏移量，正是这个偏移量才让卷积变形为不规则的卷积，这里要注意这个偏移量可以是小数，所以下面的式子的特征值需要通过*双线性插值*的方法来计算。：

![v2-7a06b5893d8008a7f7219f204803da8f_720w](C:\Users\86133\Desktop\images\v2-7a06b5893d8008a7f7219f204803da8f_720w.jpg)

那这个偏移量如何算呢？我们来看：

![v2-25ebc589a204289291645f57dcf92314_720w](C:\Users\86133\Desktop\images\v2-25ebc589a204289291645f57dcf92314_720w.jpg)



对于输入的一张feature map，假设原来的卷积操作是3×3的，那么为了学习偏移量offset，我们定义另外一个3×3的卷积层（图中上面的那层），输出的维度其实就是原来feature map大小，channel数等于2N（分别表示x,y方向的偏移）。下面的可变形卷积可以看作先基于上面那部分生成的offset做了一个插值操作，然后再执行普通的卷积。

v2:

背景：

DCN v1听起来不错，但其实也有问题：我们的可变形卷积有可能引入了无用的上下文（区域）来干扰我们的特征提取，这显然会降低算法的表现

所以作者提出了三个解决方法：

（1）More Deformable Conv Layers（使用更多的可变形卷积）

（2）Modulated Deformable Modules（在DCNv1基础（添加offset）上添加每个采样点的权重）

（3）R-CNN Feature Mimicking（模拟R-CNN的feature）

使用更多的可变形卷积：

在DCN v1中只在conv 5中使用了三个可变形卷积，在DCN v2中把conv3到conv5都换成了可变形卷积，提高算法对几何形变的建模能力。

![img](C:\Users\86133\Desktop\images\v2-a4e0056b833385e2ea37522f89a86d7d_720w.jpg)

在DCNv1基础（添加offset）上添加每个采样点的权重

我们知道在DCN v1中的卷积是添加了一个offset![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7BP_n%7D):

![v2-7a06b5893d8008a7f7219f204803da8f_720w (1)](C:\Users\86133\Desktop\images\v2-7a06b5893d8008a7f7219f204803da8f_720w (1).jpg)

为了解决引入了一些无关区域的问题，在DCN v2中我们不只添加每一个采样点的偏移，还添加了一个权重系数![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7Bm_k%7D)，来区分我们引入的区域是否为我们感兴趣的区域，假如这个采样点的区域我们不感兴趣，则把权重学习为0即可：

![v2-4d7662db5e10d0c857a57e9cc24bcfd4_720w](C:\Users\86133\Desktop\images\v2-4d7662db5e10d0c857a57e9cc24bcfd4_720w.jpg)

*总的来说，DCN v1中引入的offset是要寻找有效信息的区域位置，DCN v2中引入权重系数是要给找到的这个位置赋予权重，这两方面保证了有效信息的准确提取*。
