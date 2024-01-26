1.损失函数补充：Huber Loss，也被称作Smooth Mean Abusolute Error Loss。
均方差损失MSE和平均绝对误差损失MAE是两种常用的损失函数，分别对应的概率分布假设为高斯分布和拉普拉斯分布。这两者对比的话，MSE比MAE能够更快收敛。MSE的梯度能够随着误差大小发生变化，而MAE损失的梯度总是正负1，这不利于模型的训练。但是MAE对异常点更加鲁棒。从函数的角度看，MSE对误差平方化，使得异常点的误差会特别大，从原理的角度看，拉普拉斯分布也对异常点更鲁棒。
Huber Loss则是将两者的优点结合起来的损失函数。其原理很简单，就是在误差接近0时使用MSE， 在误差较大时使用MAE，公式为：
![](https://ai-studio-static-online.cdn.bcebos.com/fd78d8c926614d8cb230f8338f0cee5cf7ad22e956bc488ea421cf7b27888573)
1
上式中的δ是该损失函数的超参数，它的值是MSE和MAE相连接的位置。在正负δ范围内就是MSE，使损失函数可到且梯度更加稳定，范围之外则是MAE,降低了异常点的影响，使训练更加鲁棒。


```python
import numpy as np
#简单实现,设y，z是一维的
def Huber_Loss (y,z,e):
    m = 0
    N = len(y)
    for i in range(N):
        if y[i]-z[i] <=e :
            m += pow(y[i]-z[i], 2)/2
        else :
            m += np.absolute(e*(y[i]-z[i])) - e*e/2
    return m/N

```

3.池化方法补充：
对于一般的池化方法，补充：
随机池化。随机池化会对模板内的元素生成对应的概率矩阵，模板内元素值越大的概率就越大。这种池化方法不会一直选择最大值，但是它的效果不稳定，不能保证池化的结果一定是好的。
![](https://ai-studio-static-online.cdn.bcebos.com/651a3bdd3bc444e385256404610a5d768859e70f02a7411d978c90da626fe764)
2
重叠池化：和一般的池化方法不同。重叠池化的步长和模板大小不同，两个池化区域存在重叠。经过实验，这种池化方法能够少量提高准确率。
![](https://ai-studio-static-online.cdn.bcebos.com/77ef8b7dda994392965d74cf3969d1ffd5426d7b801c49c4b3a4de8d1c2bbb85)
3
金字塔池化：一般CNN对输入图像的尺寸有要求，这是因为全连接层的神经元个数对输入的特征维度是固定的。但采用金字塔池化，则可以将任意图像的卷积特征图像转化为所指定维度的特征向量输入给全连接层，这就使得CNN输入图像尺寸可以是任意的。空间金字塔池化是将池化层转化为多尺度的池化，即利用多个大小尺寸不同的池化模板进行池化操作。
![](https://ai-studio-static-online.cdn.bcebos.com/f010a6c5e40149af8583c2260052b6da2d62d6585eff4a1d98d39f7d15122f82)
4


4.数据增强方法修改和补充
Color Jittering: 对颜色的数据增强，使得图像的亮度，饱和度，对比度发生变化，但是有可能导致反作用。
PCA Jittering: 按照通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值，再来做PCA Jittering.
Noise :对图像加噪处理，可以加入高斯噪声，椒盐噪声等。但是高斯噪声可能具有较低的信息失真水平。
还可以训练GAN网络来进行数据增强，还有学习增广，有监督数据增广，Label,shuffle等方法。

5.图像分类方法综述
1.使用神经网络进行图像分类
深度学习和神经网络自从AlexNet在ImagNet上取得优秀成果之后就在图像处理领域迅速发展，在这一领域的很多分支都已经有对应的神经网络了。
用于图像分类的神经网络就有LeNet， AlexNet, VGG,  GoogLeNet, DarkNet, ResNet, ViT这些网络，还有许多它们的性能更加优秀的变体。
2.传统方法
传统的用于图像分类的算法有KNN, 基于一些技术的SVM等方法。
