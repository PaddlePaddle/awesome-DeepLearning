**1.损失函数方法补充**

Loss函数需要满足一些性质，首先最明显的，它应该能够描述我们上面概念中的"接近"。预测值接近标签 Loss 应该小，否则 Loss 应该大。 其次，深度学习使用的梯度下降优化方法需要让模型权重沿着 Loss 导数的反向更新，所以这个函数需要是可导的，而且不能导数处处为0(这样权重动不了)。

不同的深度学习任务输出的形式不同，定义 “接近” 的标准也不同，Loss 因此有很多种，但大致可以分为两类: 回归 Loss 和 分类 Loss。 回归和分类两类问题最明显的区别是： 回归预测的结果是连续的（比如房价），而分类预测的结果是离散的（比如手写识别输出这个数字是几）。

①均方误差、平方损失——L2损失：

均方误差（MSE）是回归损失函数中最常用的误差，它是预测值与目标值之间差值的平方和

![](https://ai-studio-static-online.cdn.bcebos.com/0d4dcca5dcb2495cbf3d26a856d4fdca54a1d47cabf44c409cd78263b81b78af)



②平均绝对误差——L1损失函数：

平均绝对误差（MAE）是另一种常用的回归损失函数，它是目标值与预测值之差绝对值的和，表示了预测值的平均误差幅度，而不需要考虑误差的方向（注：平均偏差误差MBE则是考虑的方向的误差，是残差的和），范围是0到∞。

![](https://ai-studio-static-online.cdn.bcebos.com/8ee6eae312634cea8f548cc6ac1ea1563db7fb52aa7f49a38683362b8ea25648)


③Huber损失——平滑平均绝对误差：

Huber损失相比于平方损失来说对于异常值不敏感，但它同样保持了可微的特性。它基于绝对误差但在误差很小的时候变成了平方误差。我们可以使用超参数δ来调节这一误差的阈值。当δ趋向于0时它就退化成了MAE，而当δ趋向于无穷时则退化为了MSE，其表达式如下，是一个连续可微的分段函数：

![](https://ai-studio-static-online.cdn.bcebos.com/de1622506a4a4d05a6578933723679f667a8cad9ba7b433e95b13a188bb14fe9)


④Log-Cosh损失函数

Log-Cosh损失函数是一种比L2更为平滑的损失函数，利用双曲余弦来计算预测误差：

![](https://ai-studio-static-online.cdn.bcebos.com/80019bc8d99f43698d1047757a3f8b273b24309b46ca4a9f834d90d4ddc72ade)




**2.损失函数python代码实现**

```
def mse(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: mean square error loss
    """
    
    return np.sum((true - pred)**2)
```

```
def mae(true, pred):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: mean absolute error loss
    """
    
    return np.sum(np.abs(true - pred))
```

```
def sm_mae(true, pred, delta):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)
```

```
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)
```

    


**3.池化方法**

filter(特征抽取器，卷积核，CV上称之为滤波器）在一个窗口（text region）上可以抽取出一个特征值，filter在整个text上滑动，将抽取出一系列特征值组成一个特征向量。这就是卷积层抽取文本特征的过程。模型中的每一个filter都如此操作，形成了不同的特征向量。

pooling层则对filters的抽取结果进行降维操作，获得样本的重要特征，为下一次的卷积增加感受野的大小，逐渐减小＂分辨率＂, 为最后的全连接做准备。pooling层是CNN中用来减小尺寸，提高运算速度的，同样能减小噪声的影响，让各特征更具有健壮性。降维操作方式的不同产生不同的池化方法。

一般在pooling层之后连接全连接神经网络，形成最后的分类结果。


**①Max Pooling**

对于某个filter抽取到若干特征值，只取其中得分最大的那个值作为pooling层保留值，其它特征值全部抛弃，值最大代表只保留这些特征中最强的，而抛弃其它弱的此类特征。

优点

只保留区域内的最大值（特征），忽略其它值，降低噪声的影响，提高模型健壮性;

Max Pooling能减少模型参数数量，有利于减少模型过拟合问题。因为经过pooling操作后，在NLP任务中往往把一维的数组转换为单一数值，这样对于后续的卷积层或者全联接隐层来说无疑单个filter的参数或者隐层神经元个数就减少了。

Max Pooling可以把变长的输入X整理成固定长度的输入。因为CNN最后往往会接全联接层，而其神经元个数是需要事先定好的，如果输入是不定长的那么很难设计网络结构。在NLP任务中，文本的长度往往是不确定的，而通过pooling 操作，每个filter固定取1个值，那么有多少个filter，pooling层就有多少个神经元(pooling层神经元个数等于filters个数)，这样就可以把全联接层神经元个数固定住。

**②K-Max Pooling**

K-Max Pooling可以取每一个filter抽取的一些列特征值中得分在前K大的值，并保留他们的相对的先后顺序。把所有filters的前k大的特征值拼接成一个特征向量。pooling层的神经元个数等于k倍的filter个数。就是说通过多保留一些特征信息供后续阶段使用。

优点

K-Max Pooling可以表达同一类特征出现多次的情形，即可以表达某类特征的强度；

因为这些Top-K特征值的相对顺序得以保留，所以应该说其保留了部分位置信息。


**③Chunk-Max Pooling**

把某个filter抽取到的特征向量进行分段，切割成若干段后，在每个分段里面各自取得一个最大特征值，比如将某个filter的特征向量切成3个chunk，那么就在每个chunk里面取一个最大值，于是获得3个特征值。

优点

Chunk-Max Pooling可以保留了多个局部最大特征值的相对顺序信息；

如果多次出现强特征，Chunk-Max Pooling可以捕获特征强度。


**4.数据增强方法**

数据增强也叫数据扩增，意思是在不实质性的增加数据的情况下，让有限的数据产生等价于更多数据的价值。

数据增强可以分为，有监督的数据增强和无监督的数据增强方法。其中有监督的数据增强又可以分为单样本数据增强和多样本数据增强方法，无监督的数据增强分为生成新的数据和学习增强策略两个方向。

**有监督的数据增强**：即采用预设的数据变换规则，在已有数据的基础上进行数据的扩增，包含单样本数据增强和多样本数据增强，其中单样本又包括几何操作类，颜色变换类。

单样本数据增强：所谓单样本数据增强，即增强一个样本的时候，全部围绕着该样本本身进行操作，包括几何变换类，颜色变换类等。

多样本数据增强：不同于单样本数据增强，多样本数据增强方法利用多个样本来产生新的样本（SMOTE、SamplePairing、mixup）

**无监督的数据增强**：无监督的数据增强方法包括两类：

通过模型学习数据的分布，随机生成与训练数据集分布一致的图片，代表方法GAN。

通过模型，学习出适合当前任务的数据增强方法，代表方法AutoAugment。



**5.图像分类方法综述**

图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉的核心，是物体检测、图像分割、物体跟踪、行为分析、人脸识别等其他高层次视觉任务的基础。图像分类在许多领域都有着广泛的应用，如：安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。

**传统方法**：传统图像分类通过手工提取特征或特征学习方法对整个图像进行全部描述，然后使用分类器判别物体类别，因此如何提取图像的特征至关重要。

特征提取：主要包括纹理、颜色、形状等底层视觉特征，尺度不变特征变换、局部二值模式、方向梯度直方图等局部不变性特征。

分类器：主要包括kNN(k-nearest neighbor,k最近邻）决策树、SVM(support vector machine，支持向量机）、人工神经网络等方法。

**深度学习方法**：深度学习是机器学习的一种新兴算法，因其在图像特征学习方面具有显著效果而受到研究者们的广泛关注。相较于传统的图像分类方法，其不需要对目标图像进行人工特征描述和提取，而是通过神经网络自主地从训练样本中学习特征，提取出更高维、抽象的特征，并且这些特征与分类器关系紧密，很好地解决了人工提取特征和分类器选择的难题，是一种端到端的模型。

常用的标准网络模型：Lenet、Alxnet、Vgg系列、Resnet系列、Inception系列、Densenet系列、Googlenet、Nasnet、Xception、Senet(state of art)；

轻量化网络模型：Mobilenet v1,v2、Shufflenet v1,v2,Squeezenet

