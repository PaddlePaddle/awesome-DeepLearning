###  1.深度学习基础知识(10分每题)

#### ①损失函数方法补充∶补充目前awesome-DeepLearning repo缺少的损失函数，进行算法补充

**MaxOut函数**

这个函数可以参考论文《maxout networks》，Maxout是深度学习网络中的一层网络，就像池化层、卷积层一样等，我们可以把Maxout 看成是网络的激活函数层，我们假设网络某一层的输入特征向量为：X=（x1,x2,……xd），也就是我们输入是d个神经元。Maxout隐藏层每个神经元的计算公式如下：

![](https://latex.codecogs.com/svg.image?h_i(x)=MAXz_{ij},j\in[1,k])

上面的公式就是Maxout隐藏层神经元i的计算公式。其中，k就是Maxout层所需要的参数了，由我们人为设定大小。就像dropout一样，也有自己的参数p(每个神经元dropout概率)，Maxout的参数是k。公式中Z的计算公式为：

<img src="https://latex.codecogs.com/svg.image?z_{ij}=x^TW..._{ij}&space;&plus;&space;b_{ij}" title="z_{ij}=x^TW..._{ij} + b_{ij}" />

权重w是一个大小为(d,m,k)三维矩阵，b是一个大小为(m,k)的二维矩阵，这两个就是我们需要学习的参数。如果我们设定参数k=1，那么这个时候，网络就类似于以前我们所学普通的MLP网络。

#### ②损失函数python代码实现∶基于python实现上述损失函数，和对应损失函数写一个markdown即可

```Python
output = K.max(K.dot(X, self.W) + self.b, axis=1)#maxout激活函数
```



#### ③池化方法补充︰同①

随机池化（stochastic pooling）

只需要对Feature Map中的元素按照其概率值大小随机选择，元素选中的概率与其数值大小正相关，并非如同max pooling那样直接选取最大值。这种随机池化操作不但最大化地保证了取值的Max，也部分确保不会所有的元素都被选取max值，从而提高了泛化能力。stochastic pooling方法非常简单，只需对feature map中的元素按照其概率值大小随机选择，即元素值大的被选中的概率也大。而不像max-pooling那样，永远只取那个最大值元素。

计算过程:

1）先将方格中的元素同时除以它们的和sum，得到概率矩阵；

2）按照概率随机选中方格；

3）pooling得到的值就是方格位置的值。

使用stochastic pooling时(即test过程)，其推理过程也很简单，对矩阵区域求加权平均即可。

在反向传播求导时，只需保留前向传播已经记录被选中节点的位置的值，其它值都为0,这和max-pooling的反向传播非常类似。

#### ④数据增强方法修改及补充︰同①

几何变换方法主要有：翻转，旋转，裁剪，缩放，平移，抖动。

像素变换方法有：加椒盐噪声，高斯噪声，进行高斯模糊，调整 HSV 对比度，调节亮度，饱和度，直方图均衡化，调整白平衡等。

无监督的数据增强：GAN生成对抗网络

#### ⑤图像分类方法综述∶调研目前分类方法综述，预备下节课知识

目前图像分类都是基于深度学习神经网络来开发的。如：LeNet，AlexNet，VGG，GoogLeNet，ResNet，ViT等等

