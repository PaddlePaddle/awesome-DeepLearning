```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# （一）损失函数方法补充：
1、Huber loss
既然 MSE 和 MAE 各有优点和缺点，那么有没有一种激活函数能同时消除二者的缺点，集合二者的优点呢？答案是有的。Huber Loss 就具备这样的优点，其公式如下：

![](https://ai-studio-static-online.cdn.bcebos.com/030eb1ef82a4418ba792c3e7e869db6d2a43fc988faa4f8989a220adc39a5228)

2、Log-Cosh

Log-Cosh是应用于回归任务中的另一种损失函数，它比L2损失更平滑。Log-cosh是预测误差的双曲余弦的对数。

![](https://ai-studio-static-online.cdn.bcebos.com/e5ad13eaed974d88812b6240601142d30bc53d1a7b6a44bcb0db40ff122280cf)

3、hinge loss损失函数

![](https://ai-studio-static-online.cdn.bcebos.com/483175ca2b8244d392d36d357c916686c0d59a0a2f8f4e7889a14505f086d142)



# **（二）Huber 损失函数详解**
![](https://ai-studio-static-online.cdn.bcebos.com/b834d7d5165646959962d890d130649cb87084148be5460f966cfcd6906643a5)



```python
# Huber损失函数
import numpy as np
x = 1
X = np.vstack((np.ones_like(x),x))    # 引入常数项 1
m = X.shape[1]
# 参数初始化
W = np.zeros((1,2))
 
# 迭代训练 
num_iter = 20
lr = 0.01
delta = 2
J = []
for i in range(num_iter):
   y_pred = W.dot(X)
   loss = 1/m * np.sum(np.abs(y-y_pred))
   J.append(loss)
   mask = (y-y_pred).copy()
   mask[y-y_pred > delta] = delta
   mask[mask < -delta] = -delta
   W = W + lr * 1/m * mask.dot(X.T)
 
# 作图
y1 = W[0,0] + W[0,1]*1
y2 = W[0,0] + W[0,1]*20
plt.scatter(x, y)
plt.plot([1,20],[y1,y2],'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('MAE')
plt.show()

 
# log cosh 损失
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))return np.sum(loss)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-84d4643d5d46> in <module>
         13 for i in range(num_iter):
         14    y_pred = W.dot(X)
    ---> 15    loss = 1/m * np.sum(np.abs(y-y_pred))
         16    J.append(loss)
         17    mask = (y-y_pred).copy()


    NameError: name 'y' is not defined


# **（三）池化方法补充**

1.全局平均池化

![](https://ai-studio-static-online.cdn.bcebos.com/4905da59778a4af0b3adfb6c880dd92f3e0e5fbbf72e47d4a95d7b3ac91d83dd)

2.Max Pooling Over Time池化

![](https://ai-studio-static-online.cdn.bcebos.com/6a2980cfcd664e4f803acaee19b1a44a9ef80d9c98d847f09284ea2367b86337)


# **（四）数据增强方法修改及补充**
1。常用的方法
比较常用的几何变换方法主要有：翻转，旋转，裁剪，缩放，平移，抖动。

值得注意的是，在某些具体的任务中，当使用这些方法时需要主要标签数据的变化，如目标检测中若使用翻转，则需要将 gt 框进行相应的调整。

比较常用的像素变换方法有：加椒盐噪声，高斯噪声，进行高斯模糊，调整 HSV 对比度，调节亮度，饱和度，直方图均衡化，调整白平衡等。

2.Coutout（2017）
该方法来源于论文《Improved Regularization of Convolutional Neural Networks with Cutout》
在一些人体姿态估计，人脸识别，目标跟踪，行人重识别等任务中常常会出现遮挡的情 况，为了提高模型的鲁棒性，提出了使用 Cutout 数据增强方法。该方法的依据是 CutOut 能 够让 CNN 更好地利用图像的全局信息，而不是依赖于一小部分特定的视觉特征。
做法：对一张图像随机选取一个小正方形区域，在这个区域的像素值设置为 0 或其它统 一的值。
注：存在 50%的概率不对图像使用 Cutout。
效果图如下：

![](https://ai-studio-static-online.cdn.bcebos.com/d5d681de7c054c4eba308d2474554f6569fc9a122c8a44d3bbf3b472e4aede37)

3.Random Erasing该方法来源于论文《Random Erasing Data Augmentation》。 这个方法有点类似于 Cutout，这两者同一年发表的。与 Cutout 不同的是，Random Erasing 掩码区域的长宽，以及区域中像素值的替代值都是随机的，Cutout 是固定使用正方形，替代 值都使用同一个。 具体算法如下：

![](https://ai-studio-static-online.cdn.bcebos.com/21998c3ae3d74099b8a430cdc8b0529ab9fd774d2a6047c286ced045dc0d097d)


# **（五）图像分类方法综述**

摘要：图像物体分类与检测是计算机视觉研究中的两个重要的基本问题，也是图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。
本文从物体分类与检测问题的基本定义出发，首先从实例、类别、语义三个层次对物体分类与检测研究中存在的困难与挑战进行了阐述。
接下来，本文以物体检测和分类方面的典型数据库和国际视觉竞赛PASCAL VOC为主线对近年来物体分类与检测的发展脉络进行了梳理与总结，指出表达学习和结构学习分别对于物体分类与检测的作用。
最后本文对物体分类与检测的发展方向进行了思考和讨论，探讨了这一领域下一步研究的方向。

关键词 物体分类 物体检测 计算机视觉 特征表达 结构学习

物体分类与检测是计算机视觉、模式识别与机器学习领域非常活跃的研究方向。物体分类与检测在很多领域得到广泛应用，包括安防领域的人脸识别、行人检测、智能视频分析、行人跟踪等，交通领域的交通场景物体识别、车辆计数、逆行检测、车牌检测与识别，以及互联网领域的基于内容的图像检索、相册自动归类等。

可以说，物体分类与检测已经应用于人们日常生活的方方面面，计算机自动分类与检测技术也在一定程度了减轻了人的负担，改变了人类生活方式。计算机视觉理论的奠基者，英国神经生理学家马尔认为，视觉要解决的问题可归结为“What is Where”，即什么东西在什么地方，即计算机视觉的研究中，物体分类和检测是最基本的研究问题之一。

![](https://ai-studio-static-online.cdn.bcebos.com/fc9bd0e7ad2e495eaccdc1818b344cb9bf2ad33a60e842eaa6dea822a77d3d44)

如图 1 所示，给定一张图片，物体分类要回答的问题是这张图片中是否包含某类物体（比如牛）；物体检测要回答的问题则是物体出现在图中的什么地方，即需要给出物体的外接矩形框，如图 1(b)所示。

物体分类与检测的研究，是整个计算机视觉研究的基石，是解决跟踪、分割、场景理解等其他复杂视觉问题的基础。欲对实际复杂场景进行自动分析与理解，首先就需要确定图像中存在什么物体（分类问题），或者是确定图像中什么位置存在什么物体(检测问题)。鉴于物体分类与检测在计算机视觉领域的重要地位，研究鲁棒、准确的物体分类与检测算法，无疑有着重要的理论意义和实际意义。

物体分类与检测的难点与挑战
物体分类与检测是视觉研究中的基本问题，也是一个非常具有挑战性的问题。物体分类与检测的难点与挑战在本文中分为三个层次：实例层次、类别层次、语义层次，如图 2 所示。

![](https://ai-studio-static-online.cdn.bcebos.com/1716ebf98ba74f6f875b8bfb51e4ba2e1d96fff64d1e43558ec32acd849938ec)

实例层次:
针对单个物体实例而言，通常由于图像采集过程中光照条件、拍摄视角、距离的不同，物体自身的非刚体形变以及其他物体的部分遮挡使得物体实例的表观特征产生很大的变化，给视觉识别算法带来了极大的困难。

类别层次:
困难与挑战通常来自三个方面，首先是类内差大，也即属于同一类的物体表观特征差别比较大，其原因有前面提到的各种实例层次的变化，但这里更强调的是类内不同实例的差别，例如图 3(a)所示，同样是椅子，外观却是千差万别，而从语义上来讲，有“坐”的功能的器具都可以称椅子；其次是类间模糊性，即不同类的物体实例具有一定的相似性，如图 3(b)所示，左边的是一只狼，右边的是一只哈士奇，但我们从外观上却很难分开二者；再次是背景的干扰，在实际场景下，物体不可能出现在一个非常干净的背景下，往往相反，背景可能是非常复杂的、对我们感兴趣的物体存在干扰的，这使得识别问题的难度大大加大。
本文从物体分类与检测问题的基本定义出发，首先从实例、类别、语义三个层次对物体分类与检测研究中存在的困难与挑战进行了阐述。
接下来，本文以物体检测和分类方面的主流数据库和国际视觉竞赛 PASCAL VOC 竞赛为主线对近年来物体分类与检测算法的发展脉络进行了梳理与总结，指出物体分类与检测算法的主流方法：基于表达学习和结构学习。在此基础上，本文对物体分类与检测算法的发展方向进行了思考和讨论，指出了物体检测和物体分类的有机统一，探讨了下一步研究的方向。

语义层次:
困难与挑战与图像的视觉语义相关，这个层次的困难往往非常难以处理，特别是对现在的计算机视觉理论水平而言。一个典型的问题称为多重稳定性。如图所示，图 3(c)左边既可以看成是两个面对面的人，也可以看成是一个燃烧的蜡烛；右边则同时可以解释为兔子或者小鸭。同样的图像，不同的解释，这既与人的观察视角、关注点等物理条件有关，也与人的性格、经历等有关，而这恰恰是视觉识别系统难以很好处理的部分。

**物体分类与检测数据库**

数据是视觉识别研究中最重要的因素之一，通常我们更多关注于模型、算法本身，事实上，数据在视觉任务的作用正越来越明显，大数据时代的到来，也使得研究人员开始更加重视数据。在数据足够多的情况下，我们甚至可以使用最简单的模型、算法，比如最近邻分类，朴素贝叶斯分类器都能得到很好的效果。鉴于数据对算法的重要性，我们将在本节对视觉研究中物体分类与检测方面的主流数据进行概述，从中也可以一窥目标分类、检测的发展。在介绍不同数据库时，将主要从数据库图像数目、类别数目、每类样本数目、图像大小、分类检测任务难度等方面进行阐述，如表 3 中所示。
早期物体分类研究集中于一些较为简单的特定任务，如 OCR、形状分类等。OCR 中数字手写识别是一个得到广泛研究的课题，相关数据库中最著名的是 MNIST数据库。MNIST 是一个数字手写识别领域的标准评测数据集，数据库大小是60000，一共包含 10 类阿拉伯数字，每类提供 5000张图像进行训练，1000 张进行测试。MNIST 的图像大小为 28×28，即 784 维，所有图像手写数字，存在较大的形变。形状分类是另一个比较重要的物体分类初期的研究领域，相关数据库有 ETHZ Shape Classes，MPEG-7等。其中 ETHZ ShapeClasses 包含 6 类具有较大差别的形状类别：苹果、商标、瓶子、长颈鹿、杯子、天鹅，整个数据库包含 255 张测试图像。

CIFAR-10&CIFAR-100 数 据 库 是 Tiny images的两个子集，分别包含了 10 类和 100 类物体类别。这两个数据库的图像尺寸都是 32×32，而且是彩色图像。CIFAR-10 包含 6 万的图像，其中 5 万用于模型训练，1 万用于测试，每一类物体有 5000 张图像用于训练，1000 张图像用于测试。

CIFAR-100 与 CIFAR-10 组成类似，不同是包含了更多的类别：20 个大类，大类又细分为 100 个小类别，每类包含 600 张图像。CIFAR-10 和 CIFAR-100数据库尺寸较小，但是数据规模相对较大，非常适合复杂模型特别是深度学习模型训练，因而成为深度学习领域主流的物体识别评测数据集。

Caltech-101是第一个规模较大的一般物体识别标准数据库，除背景类别外，它一共包含了 101类物体，共 9146 张图像，每类中图像数目从 40 到800 不等，图像尺寸也达到 300 左右。Caltech-101是以物体为中心构建的数据库，每张图像基本只包含一个物体实例，且居于图像中间位置，物体尺寸相对图像尺寸比例较大，且变化相对实际场景来说不大，比较容易识别。Caltech-101 每类的图像数目
差别较大，有些类别只有很少的训练图像，也约束了 可以使 用的训 练集大小 。

Caltech 256 与Caltech-101 类似，区别是物体类别从 101 类增加到了 256 类，每类包含至少 80 张图像。图像类别的增加，也使得 Caltech-256 上的识别任务更加困难，使其成为检验算法性能与扩展性的新基准。15Scenes 是由 Svetlana Lazebnik 在 FeiFei Li 的13 Scenes 数据库的基础上加入了两个新的场景构成的，一共有 15 个自然场景，4485 张图像，每类大概包含 200 到 400 张图像，图像分辨率约为300 × 250。15 Scenes 数据库主要用于场景分类评测，由于物体分类与场景分类在模型与算法上差别不大，该数据库也在图像分类问题上得到广泛的使用。

PASCAL VOC 从 2005 年到 2012 年每年发布关于分类、检测、分割等任务的数据库，并在相应数据库上举行了算法竞赛，极大地推动了视觉研究的发展进步。最初 2005 年 PASCAL VOC 数据库只包含人、自行车、摩托车、汽车 4 类，2006 年类别数目增加到 10 类，2007 年开始类别数目固定为 20 类，以后每年只增加部分样本。PASCAL VOC数据库中物体类别均为日常生活常见的物体，如交通工具、室内家具、人、动物等。PASCAL VOC 2007数据库共包含 9963 张图片，图片来源包括 Filker等互联网站点以及其它数据库，每类大概包含96-2008 张图像，均为一般尺寸的自然图像。PASCAL VOC 数据库与 Caltech-101 相比，虽然类别数更少，但由于图像中物体变化极大，每张图像可能包含多个不同类别物体实例，且物体尺度变化很大，因而分类与检测难度都非常大。该数据库的提出，对物体分类与检测的算法提出了极大的挑战，也催生了大批优秀的理论与算法，将物体识别研究推向了一个新的高度。

随着分类与检测算法的进步，很多算法在以上提到的相关数据库上性能都接近饱和，同时随着大数据时代的到来，硬件技术的发展，也使得在更大规 模 的 数 据 库 进 行 研 究 和 评 测 成 为 必 然 。

ImageNet是由 FeiFei Li 主持构建的大规模图像数据库，图像类别按照 WordNet 构建，全库截至2013 年共有 1400 万张图像，2.2 万个类别，平均每类包含 1000 张图像。这是目前视觉识别领域最大的有标注的自然图像分辨率的数据集，尽管图像本身基本还是以目标为中心构建的，但是海量的数据和海量的图像类别，使得该数据库上的分类任务依然极具挑战性。除此之外，ImageNet 还构建了一个包含 1000 类物体 120 万图像的子集，并以此作为ImageNet 大尺度视觉识别竞赛的数据平台，逐渐成为物体分类算法评测的标准数据集。

SUN 数据库的构建是希望给研究人员提供一个覆盖较大场景、位置、人物变化的数据库，库中的场景名是从 WordNet 中的所有场景名称中得来。SUN 数据库包含两个评测集，一个是场景识别数据集，称为 SUN-397，共包含 397 类场景，每类至少包含 100 张图片，总共有 108,754 张图像。另一个评测集为物体检测数据集，称为 SUN2012,包含 16,873 张图像。Tiny images是一个图像规模更大的数据库，共包含 7900 万张 32×32 图像，图像类别数目有 7.5 万，尽管图像分辨率较低，但还是具有较高的区分度，而其绝无仅有的数据规模，使其成为大规模分类、检索算法的研究基础。我们通过分析表 1 可以看到，物体分类的发展过程中，数据库的构建大概可以分为 3 个阶段，经历了一个从简单到复杂，从特殊到一般，从小规模到大规模的跨越。

早期的手写数字识别 MNIST，形状分类 MPEG-7 等都是研究特定问题中图像分类，之后研究人员开始进行更广泛的一般目标分类与检 测 的 研 究 ， 典 型 的 数 据 库 包 括 15 场 景 ，Caltech-101/256, PASCAL VOC 2007 等；随着词包模型等算法的发展与成熟，更大规模的物体分类与检测研究得到了广泛的关注，这一阶段的典型数据库包括 SUN 数据库、ImageNet 以及 Tiny 等。
近年来，数据库构建中的科学性也受到越来越多的关注，Torralba 等人对数据库的 Bias、泛化性能、价值等问题进行了深入的讨论，提出排除数据库构建过程中的选择偏好，拍摄偏好，负样本集偏好是构造更加接近真实视觉世界的视觉数据库中的关键问题。伴随着视觉处理理论的进步，视觉识别逐渐开始处理更加真实场景的视觉问题，因而对视觉数据库的泛化性、规模等也提出了新的要求和挑战。
我们也可以发现，物体类别越多，导致类间差越小，分类与检测任务越困难，图像数目、图像尺寸的大小，则直接对算法的可扩展性提出了更高的要求，如何在有限时间内高效地处理海量数据、进行准确的目标分类与检测成为当前研究的热点。

**物体分类与检测发展历程**

图像物体识别的研究已经有五十多年的历史。各类理论和算法层出不穷，在这部分，我们对物体分类与检测的发展脉络进行了简单梳理，并将其中里程碑式的工作进行综述。特别的，我们以国际视觉竞赛 PASCAL VOC竞赛为主线对物体分类与检测算法近年来的主要进展进行综述，这个系列的竞赛对物体分类检测的发展影响深远，其工作也代表了当时的最高水平。

物体分类 任务要求回答一张图像中是否包含某种物体，对图像进行特征描述是物体分类的主要研究内容。一般说来，物体分类算法通过手工特征或者特征学习方法对整个图像进行全局描述，然后使用分类器判断是否存在某类物体。

物体检测 任务则更为复杂，它需要回答一张图像中在什么位置存在一个什么物体，因而除特征表达外，物体结构是物体检测任务不同于物体分类的最重要之处。总的来说，近年来物体分类方法多侧重于学习特征表达，典型的包括词包模型(Bag-of-Words)、深度学习模型；物体检测方法则侧重于结构学习，以形变部件模型为代表。

这里我们首先以典型的分类检测模型来阐述其一般方法和过程，之后以 PASCAL VOC（包含 ImageNet)竞赛历年来的最好成绩来介绍物体分类和物体检测算法的发展，包括物体分类中的词包模型、深度学习模型以及物体检测中的结构学习模型，并分别对各个部分进行阐述。

1. 底层特征提取

底层特征是物体分类与检测框架中的第一步，底层特征提取方式有两种：一种是基于兴趣点检测，另一种是采用密集提取的方式。

兴趣点检测算法通过某种准则选择具有明确定义的、局部纹理特征比较明显的像素点、边缘、角点、区块等，并且通常能够获得一定的几何不变性，从而可以在较小的开销下得到更有意义的表达，最常用的兴趣点检测算子有 Harris 角点检测子、FAST(Features from Accelerated Segment Test) 算子、LoG (Laplacian of Gaussian)、DoG (Difference ofGaussian)等。近年来物体分类领域使用更多的则是密集提取的方式，从图像中按固定的步长、尺度提取出大量的局部特征描述，大量的局部描述尽管具有更高的冗余度，但信息更加丰富，后面再使用词包模型进行有效表达后通常可以得到比兴趣点检测 更 好 的 性 能 。
常 用 的 局 部 特 征 包 括 SIFT(Scale-invariant feature transform，尺度不变特征转换)、HOG(Histogram of Oriented Gradient, 方向梯度直方图) 、LBP(Local Binary Pattern, 局部二值模式) 等。从表 2 可以看出，历年最好的物体分类算法都采用了多种特征，采样方式上密集提取与兴趣点检测相结合，底层特征描述也采用了多种特征描述子，这样做的好处是，在底层特征提取阶段，通过提取到大量的冗余特征，最大限度的对图像进行底层描述，防止丢失过多的有用信息，这些底层描述中的冗余信息主要靠后面的特征编码和特征汇聚得到抽象和简并。事实上，近年来得到广泛关注的深度学习理论中一个重要的观点就是手工设计的底层特征描述子作为视觉信息处理的第一步，往往会过早的丢失有用的信息，直接从图像像素学习到任务相关的特征描述是比手工特征更为有效的手段。

2.特征编码

密集提取的底层特征中包含了大量的冗余与噪声，为提高特征表达的鲁棒性，需要使用一种特征变换算法对底层特征进行编码，从而获得更具区分性、更加鲁棒的特征表达，这一步对物体识别的性能具有至关重要的作用，因而大量的研究工作都集中在寻找更加强大的特征编码方法，重要的特征编码算法包括向量量化编码、核词典编码、稀疏编码、局部线性约束编码、显著性编码、Fisher 向量编码、超向量编码等。最简单的特征编码是向量量化编码，它的出现甚至比词包模型的提出还要早。向量量化编码是通过一种量化的思想，使用一个较小的特征集合（视觉词典）来对底层特征进行描述，达到特征压缩的目的。向量量化编码只在最近的视觉单词上响应为 1，因而又称为硬量化编码、硬投票编码，这意味着向量量化编码只能对局部特征进行很粗糙的重构。但向量量化编码思想简单、直观，也比较容易高效实现，因而从 2005 年第一届PASCAL VOC 竞赛以来，就得到了广泛的使用。
在实际图像中，图像局部特征常常存在一定的模糊性，即一个局部特征可能和多个视觉单词差别很小，这个时候若使用向量量化编码将只利用距离最近的视觉单词，而忽略了其他相似性很高的视觉单词。为了克服这种 模糊性问题，Gemert 等提出了软量化编码（又称核视觉词典编码）算法，局部特征不再使用一个视觉单词描述，而是由距离最近的 K 个视觉单词加权后进行描述，有效解决了视觉单词的模糊性问题，提高了物体识别的精度。稀疏表达理论近年来在视觉研究领域得到了大量的关注，研究人员最初在生理实验中发现细胞在绝大部分时间内是处于不活动状态，也即在时间轴上细胞的激活信号是稀疏的。稀疏编码通过最小二乘重构加入稀疏约束来实现在一个过完备基上响应的稀疏性。
ℓ 约束是最直接的稀疏约束，但通常很难进行优化，近年来更多使用的是 ℓ 约束，可以更加有效地进行迭代优化，得到稀疏表达。2009 年杨建超等人 将稀疏编码应用到物体分类领域，替代了之前的向量量化编码和软量化编码,得到一个高维的高度稀疏的特征表达，大大提高了特征表达的线性可分性， 仅仅使用线性分类器就得到了当时最好的物体分类结果，将物体分类的研究推向了一个新的高度上。稀疏编码在物体分类上的成功也不难理解，对于一个很大的特征集合（视觉词典），一个物体通常只和其中很少的特征有关，例如，自行车通常和表达车轮、车把等部分的视觉单词密切相关，与飞机机翼、电视机屏幕等关系很小，而行人则通常在头、四肢等对应的视觉单词上有强响应。稀疏编码存在一个问题，即相似的局部特征可能经过稀疏编码后在不同的视觉单词上产生响应，这种变换的不连续性必然会产生编码后特征的不匹配，影响特征的区分性能。
局部线性约束编码的提出就是为了解决这一问题，它通过加入局部线性约束，在一个局部流形上对底层特征进行编码重构，这样既可以保证得到的特征编码不会有稀疏编码存在的不连续问题，也保持了稀疏编码的特征稀疏性。局部线性约束编码中，局部性是局部线性约束编码中的一个核心思想，通过引入局部性，一定程度上改善了特征编码过程的连续性问题，即距离相近的局部特征在经过编码之后应该依然能够落在一个局部流形上。
局部线性约束编码可以得到稀疏的特征表达，与稀疏编码不同之处就在于稀疏编码无法保证相近的局部特征编码之后落在相近的局部流形。从表 2 可以看出，2009 年的分类竞赛冠军采用了混合高斯模型聚类和局部坐标编码（局部线性约束编码是其简化版本），仅仅使用线性分类器就取得了非常好的性能。不同于稀疏编码和局部线性约束编码，显著性编码引入了视觉显著性的概念，如果一个局部特征到最近和次近的视觉单词的距离差别很小，则认为这个局部特征是不“显著的”，从而编码后的响应也很小。显著性编码通过这样很简单的编码操作，在 Caltech 101/256, PASCAL VOC 2007 等数据库上取得了非常好的结果，而且由于是解析的结果，编码速度也比稀疏编码快很多。黄等人发现显著性表达配合最大值汇聚在特征编码中有重要的作用，并认为这正是稀疏编码、局部约束线性编码等之所以在图像分类任务上取得成功的原因。
超向量编码，Fisher 向量编码是近年提出的性能最好的特征编码方法，其基本思想有相似之处，都可以认为是编码局部特征和视觉单词的差。 Fisher 向量编码同时融合了产生式模型和判别式模型的能力，与传统的基于重构的特征编码方法不同，它记录了局部特征与视觉单词之间的一阶差分和二阶差分。超向量编码则直接使用局部特征与最近的视觉单词的差来替换之前简单的硬投票。这种特征编码方式得到的特征向量表达通常是传统基于重构编码方法的M 倍（这里 M 是局部特征的维度）。尽管特征维度要高出很多，超向量编码和 Fisher 向量编码在PASCAL VOC、ImageNet 等极具挑战性、大尺度数据库上获得了最好的性能，并在图像标注、图像分类、图像检索等领域得到应用。
2011 年 ImageNet分类竞赛冠军采用了超向量编码，2012 年 VOC 竞赛冠军则是采用了向量量化编码和 Fisher 向量编码。

3.特征汇聚

空间特征汇聚是特征编码后进行的特征集整合操作，通过对编码后的特征，每一维都取其最大值或者平均值，得到一个紧致的特征向量作为图像的特征表达。这一步得到的图像表达可以获得一定的特征不变性，同时也避免了使用特征集进行图像表达的高额代价。最大值汇聚在绝大部分情况下的性能要优于平均值汇聚，也在物体分类中使用最为广泛。由于图像通常具有极强的空间结构约束，空间金字塔匹配 (Spatial Pyramid Matching, SPM)提出将图像均匀分块，然后每个区块里面单独做特征汇聚操作并将所有特征向量拼接起来作为图像最终的特征表达。空间金字塔匹配的想法非常直观，是金字塔匹配核 (Pyramid Matching Kernel, PMK) 的图像空间对偶，它操作简单而且性能提升明显，因而在当前基于词包模型的图像分类框架中成为标准步骤。实际使用中，在Caltech 101/256 等数据库上通常使用 1×1, 2×2, 4×4的空间分块，因而特征维度是全局汇聚得到的特征向量的 21 倍，在 PASCAL VOC 数据库上，则采用1×1,2×2,3×1 的分块，因而最终特征表达的维度是全局汇聚的8倍。

4.使用支持向量机等分类器进行分类

从图像提取到特征表达之后，一张图像可以使用一个固定维度的向量进行描述，接下来就是学习一个分类器对图像进行分类。这个时候可以选择的分类器就很多了，常用的分类器有支持向量机、K 近邻、神经网络、随机森林等。基于最大化边界的支持向量机是使用最为广泛的分类器之一，在图像分类任务上性能很好，特别是使用了核方法的支持向量机。杨建超等人提出了 ScSPM 方法，通过学习过完备的稀疏特征，可以在高维特征空间提高特征的线性可分性，使用线性支持向量机就得到了当时最好的分类结果，大大降低了训练分类器的时间和空间消耗。随着物体分类研究的发展，使用的视觉单词大小不断增大，得到的图像表达维度也不断增加，达到了几十万的量级。这样高的数据维度，相比几万量级的数据样本，都与传统的模式分类问题有了很大的不同。随着处理的数据规模不断增大，基于在线学习的线性分类器成为首选，得到了广泛的关注与应用。

**深度学习**

深度学习模型是另一类物体识别算法，其基本思想是通过有监督或者无监督的方式学习层次化的特征表达，来对物体进行从底层到高层的描述 。 主 流 的 深 度 学 习 模 型 包 括 自 动 编 码 器(Auto-encoder) 、受限波尔兹曼机(Restricted Boltzmann Machine, RBM)、深度信念网络(Deep Belief Nets, DBN)、卷积神经网络(Convolutional Neural Netowrks, CNN)、生物启发式模型等。

自动编码器(Auto-encoder)是上世纪 80 年代提出的一种特殊的神经网络结构，并且在数据降维、特征提取等方面得到广泛应用。自动编码器由编码器和解码器组成，编码器将数据输入变换到隐藏层表达，解码器则负责从隐藏层恢复原始输入。隐藏层单元数目通常少于数据输入维度，起着类似“瓶颈”的作用，保持数据中最重要的信息，从而实现数据降维与特征编码。自动编码器是基于特征重构的无监督特征学习单元，加入不同的约束，可以 得 到 不 同 的 变 化 ， 包 括 去 噪 自 动 编 码 器(Denoising Autoencoders)、 稀疏 自动编 码器(Sparse Autoencoders)等，在数字手写识别、图像分类等任务上取得了非常好的结果。

受限玻尔兹曼机是一种无向二分图模型，是一种典型的基于能量的模型(Enery-based Models,EBM)。之所以称为“受限”，是指在可视层和隐藏层之间有连接，而在可视层内部和隐藏层内部不存在连接。受限玻尔兹曼机的这种特殊结构，使得它具有很好的条件独立性，即给定隐藏层单元，可视层单元之间是独立的，反之亦然。这个特性使得它可以实现同时对一层内的单元进行并行 Gibbs 采样。受限玻尔兹曼机通常采用对比散度（Contrastive Divergence，CD算法进行模型学习。受限玻尔兹曼机作为一种无监督的单层特征学习单元，类似于前面提到的特征编码算法，事实上加了稀疏约束的受限玻尔兹曼机可以学到类似稀疏编码那样的Gabor 滤波器模式。

深度信念网络(DBN)是一种层次化的无向图模型。DBN 的基本单元是 RBM（Restricted Boltzmann Machine)，首先先以原始输入为可视层，训练一个单层的RBM，然后固定第一层 RBM 权重，以 RBM 隐藏层单元的响应作为新的可视层，训练下一层的 RBM，以此类推。通过这种贪婪式的无监督训练，可以使整个 DBN 模型得到一个比较好的初始值，然后可以加入标签信息，通过产生式或者判别式方式，对整个网络进行有监督的精调，进一步改善网络性能。DBN 的多层结构，使得它能够学习得到层次化的特征表达，实现自动特征抽象，而无监督预训练过程则极大改善了深度神经网络在数据量不够时严重的局部极值问题。Hinton 等人通过这种方式，成功将其应用于手写数字识别、语音识别、基于内容检索等领域。

卷积神经网络(CNN)最早出现在上世纪80 年代，最初应用于数字手写识别，取得了一定的成功。然而，由于受硬件的约束，卷积神经网络的高强度计算消耗使得它很难应用到实际尺寸的目标识别任务上。Wisel 和 Hubel 在猫视觉系统研究工作的基础上提出了简单、复杂细胞理论，设计卷积神经网络(CNN)最早出现在上世纪80 年代，最初应用于数字手写识别，取得了一定的成功。然而，由于受硬件的约束，卷积神经网络的高强度计算消耗使得它很难应用到实际尺寸的目标识别任务上。Wisel 和 Hubel 在猫视觉系统研究工作的基础上提出了简单、复杂细胞理论，设计这里我们将最为流行的词包模型与卷积神经网络模型进行对比，发现两者其实是极为相似的。在词包模型中，对底层特征进行特征编码的过程，实际上近似等价于卷积神经网络中的卷积层，而汇聚层所进行的操作也与词包模型中的汇聚操作一样。不同之处在于，词包模型实际上相当于只包含了一个卷积层和一个汇聚层，且模型采用无监督方式进行特征表达学习，而卷积神经网络则包含了更多层的简单、复杂细胞，可以进行更为复杂的特征变换，并且其学习过程是有监督过程，滤波器权重可以根据数据与任务不断进行调整，从而学习到更有意义的特征表达。从这个角度来看，卷积神经网络具有更为强大的特征表达能力，它在图像识别任
务中的出色性能就很容易解释了。

下面我们将以 PASCAL VOC 竞赛和 ImageNet竞赛为主线，来对物体分类的发展进行梳理和分析。2005 年第一届 PASCAL VOC 竞赛数据库包含了 4 类物体：摩托车、自行车、人、汽车, 训练集加验证集一共包含 684 张图像，测试集包含 689 张图像，数据规模相对较小。从方法上来说，词包模型开始在物体分类任务上得到应用，但也存在很多其他的方法，如基于检测的物体分类、自组织网络等。从竞赛结果来看，采用“兴趣点检测-SIFT 底层特征描述-向量量化编码直方图-支持向量机”得到了最好的物体分类性能。对数线性模型和logistic 回归的性能要略差于支持向量机，这也说明了基于最大化边缘准则的支持向量机具有较强的鲁棒性，可以更好得处理物体的尺度、视角、形变等变化。
2006 年玛丽王后学院的张等人使用词包模型获得了 PASCAL VOC 物体分类竞赛冠军。与以前不同，在底层特征提取上，他们采用了更多的兴趣点检测算法，包括 Harris-Laplace 角点检测和Laplacian 块检测。除此以外，他们还使用了基于固定网格的密集特征提取方式，在多个尺度上进行特征提取。底层特征描述除使用尺度不变的 SIFT 特征外，还使用了 SPIN image 特征。
词包模型是一个无序的全局直方图描述，没有考虑底层特征的空间信息，张等人采用了 Lazebnik 提出的空间金字塔匹配方法,采用 1×1, 2×2, 3×1 的分块，因而最终特征表达的维度是全局汇聚的 8 倍。另一个与之前不同的地方在于，他们使用了一个两级的支持向量机来进行特征分类，第一级采用卡方核 SVM对空间金字塔匹配得到的各个词包特征表达进行分类，第二级则采用 RBF 核 SVM 对第一级的结果进行再分类。通过采用两级的 SVM 分类，可以将不同的 SPM 通道结果融合起来，起到一定的通道选择作用。2007 年来自 INRIA 的 Marcin Marszałek 等人获得物体分类冠军，他们所用的方法也是词包模型，基本流程与 2006 年的冠军方法类似。不同在于，他们在底层特征描述上使用了更多的底层特征描述子，包括 SIFT, SIFT-hue, PAS edgel histogram等，通过多特征方式最大可能保留图像信息，并通过特征编码和 SVM 分类方式发掘有用信息成为物体分类研究者的共识。另一个重要的改进是提出了扩展的多通道高斯核，采用学习线性距离组合的方式确定不同 SPM 通道的权重， 并利用遗传算法进行优化。
2008 年阿姆斯特丹大学和萨里大学组成的队伍获得了冠军，其基本方法依然是词包模型。
有三个比较重要的不同之处，首先是他们提出了彩色描述子来增强模型的光照不变性与判别能力；其次是使用软量化编码替代了向量量化编码，由于在实际图像中，图像局部特征常常存在一定的模糊性，即一个局部特征可能和多个视觉单词相似性差别很小，这个时候使用向量量化编码就只使用了距离最近的视觉单词，而忽略了其他同样很相似的视觉单词。为了克服这种模糊性问题，Gemert提出了软量化编码（又称核视觉词典编码）算法，有效解决了视觉模糊性问题，提高了物体识别的精度。另外，他们还采用谱回归核判别分析得到了比支持向量机更好的分类性能。2009 年物体分类研究更加成熟，冠军队伍不再专注于多底层特征、多分类器融合，而是采用了密集提取的单 SIFT 特征，并使用线性分类器进行模式分类。他们的研究中心放在了特征编码上，采用了混合高斯模型 (Gaussian Mixture Model,GMM)和局部坐标编码(Local Coordinate Coding,LCC)两种特征编码方法对底层 SIFT 特征描述子进行编码，得到了高度非线性的、局部的图像特征表达，通过提高特征的不变性、判别性来改进性能。另外，物体检测结果的融合，也进一步提升了物体分类的识别性能。局部坐标编码提出的“局部性”概念，对物体分类中的特征表达具有重要的意义 ， 之 后 出 现 的 局 部 线 性 约 束 编 码(Locality-constrained linear coding，LLC)也是基于局部性的思想，得到了“局部的”、“稀疏的”特征表达，在物体分类任务上取得了很好的结果。
2010 年冠军依旧以词包模型为基础，并且融合了物体分割与检测算法。一方面通过多底层特征、向量量化编码和空间金字塔匹配得到图像的词包模型描述，另一方面，通过使 Mean shift、过分割、基于图的分割等过分割算法，得到Patch 级的词包特征表达。这两种表达作为视觉特征表达，与检测结果以多核学习的方式进行融合。
在分类器方面，除使用了 SVM 核回归外，还提出了基于排他上下文的 Lasso 预测算法。所谓排他上下文是指一个排他标签集合中至多只能出现一种类别。排他标签集合的构建使用 Graph Shift 方法，并采用最小重构误差加稀疏约束也即 Lasso 进行预测。排他上下文作为一种不同于一般共生关系的上下文，高置信度预测可以大大抑制同一排他标签集中其他类别的置信度，改善分类性能。
2011 年冠军延续了 2010 年冠军的基本框架。来自阿姆斯特丹大学的队伍从最显著窗口对于物体分类任务的作用出发，在词包模型基础上进行了新的探索。他们发现单独包含物体的图像区域可以得到比整个图像更好的性能，一旦物体位置确定，上下文信息的作用就很小了。在物体存在较大变化的情况下，部件通常比全局更具有判别性，而在拥挤情况下，成群集合通常要比单个物体更加容易识别。基于此，他们提出了包含物体部件，整个物体，物体集合的最显著窗口框架。检测模型训练使用人工标注窗口，预测使用选择性搜索定位。词包模型和最显著窗口算法融合得到最终的分类结果。
2012 年冠军延续了 2010 年以来的算法框架，在词包模型表达方面，使用了向量量化编码、局部约束线性编码、Fisher 向量编码替代原来的单一向量量化编码。这里有两个比较重要的改进，一个是广义层次化匹配算法。考虑到传统的空间金字塔匹配算法在物体对齐的假设下才有意义，而这在实际任务中几乎不能满足，为解决这个问题，他们使用 Side 信息得到物体置信图，采用层次化的方式对局部特征进行汇聚，从而得到更好的特征匹配。另一个重要的改进是子类挖掘算法，其提出的主要目的是改进类间模糊与类内分散的问题。

**物体检测**

PASCAL VOC 竞赛从 2005 年第一届开始就引入了物体检测任务竞赛，主要任务是给定测试图片预测其中包含的物体类别与外接矩形框。物体检测任务与物体分类任务最重要的不同在于，物体结构信息在物体检测中起着至关重要的作用，而物体分类则更多考虑的是物体或者图像的全局表达。物体检测的输入是包含物体的窗口，而物体分类则是整个图像，就给定窗口而言，物体分类和物体检测在特征提取、特征编码、分类器设计方面很大程度是相通的，如表 3 所示。根据获得窗口位置策略的不同，物体检测方法大致可分为滑动窗口和广义霍夫变换两类方法。滑动窗口方法比较简单，它是通过使用训练好的模板在输入图像的多个尺度上进行滑动扫描，通过确定最大响应位置找到目标物体的外接窗口。广义霍夫投票方法则是通过在参数空间进行累加，根据局部极值获得物体位置的方法，可以用于任意形状的检测和一般物体检测任务。滑动窗口方法由于其简单和有效性，在历年的 PASCAL VOC 竞 赛 中 得 到 了 广 泛 的 使 用 。 特 别 是HOG(Histograms of Oriented Gradients)模型、形变部件模型的出现和发展，使得滑动窗口模型成为主流物体检测方法。
与物体分类问题不同，物体检测问题从数学上是研究输入图像 X 与输出物体窗口 Y 之间的关系，这里 Y 的取值不再是一个实数，而是一组“结构化”数据，指定了物体的外接窗口和类别，是一个典型的结构化学习问题。结构化支持向量机(Structrual SVM, SSVM) 基于最大化边缘准则，将普通支持向量机推广到能够处理结构化输出，有效扩展了支持向量机的应用范围，可以处理语法树、图等更一般的数据结构，在自然语言处理、机器学习、模式识别、计算机视觉等领域受到越来越多的关注。隐 变 量 支 持 向 量 机 (Latent SVM, LSVM) 是Pedro Felzenszwalb 等人在 2007 年提出用于处理物体检测问题，其基本思想是将物体位置作为隐变量放入支持向量机的目标函数中进行优化，以判别式方法得到最优的物体位置。弱标签结构化支持向量机(weak-label Structrual SVM，WL-SSVM)是一种更加一般的结构化学习框架，它的提出主要是为了处理标签空间和输出空间不一致的问题，对于多个输出符合一个标签的情况，每个样本标签都被认为是“ 弱 标 签 ”。 SSVM 和 LSVM 都 可 以 看 做 是WL-SSVM 的特例，WL-SSVM 通过一定的约简可以转化为一般的 SSVM 和 LSVM。条件随机场(Conditional Random Field, CRF)作为经典的结构化学习算法，在物体检测任务上也得到一定的关注。Schnitzspan 等人将形变部件模型与结构化学习结合，提出了一种隐条件随机场模型(latent CRFs)，通过将物体部件标签建模为隐藏节点并且采用 EM算法来进行学习，该算法突破了传统 CRF 需手动给定拓扑结构的缺点，能够自动学习到更为灵活的结构，自动发掘视觉语义上有意义的部件表达。张等提出了基于数据驱动的自动结构建模与学习来从训练数据中学习最为合适的拓扑结构。由于一般化的结构学习是一个 NP 难问题，张提出了混合结构学习方案，将结构约束分成一个弱结构项和强结构项。弱结构项由传统的树状结构模型得到，而强结构项则主要依靠条件随机场以数据驱动方式自动学习得到。

**物体检测的方法演变与发展。**

2005 年物体检测竞赛有 5 支队伍参加，采用的方法呈现多样化，Darmstadt 使用了广义霍夫变换，通过兴趣点检测和直方图特征描述方式进行特征表达，并通过广义 Hough 投票来推断物体尺度与位置，该方法在他们参加的几类中都得到了最好的性能。INRIA 的 Dalal 则采用了滑动窗口模型，底层特征使用了基于 SIFT 的描述，分类器使用支持向量机，通过采用在位置和尺度空间进行穷尽搜索，来确定物体在图像中的尺度和位置，该方法在汽车类别上取得了比广义 Hough 变换更好的性能，但在人、自行车等非刚体类别上性能并不好。
2006 年最佳物体检测算法是 Dalal 和 Triggs 提出的HOG(Histograms of Oriented Gradients)模型。他们的工作主要集中于鲁棒图像特征描述研究，提出了物体检测领域中具有重要位置的 HOG 特征。HOG 是梯度方向直方图特征，通过将图像划分成小的 Cell，在每个 Cell 内部进行梯度方向统计得到直方图描述。与 SIFT 特征相比，HOG 特征不具有尺度不变性，但计算速度要快得多。整体检测框架依然是滑动窗口策略为基础，并且使用线性分类器进行分类。这个模型本质上是一个全局刚性模板模型，需要对整个物体进行全局匹配，对物体形变不能很好地匹配处理。
2007 年 Pedro Felzenszwalb 等人提出了物体检测领域里程碑式的工作：形变部件模型(Deformable Part-based Model)，并以此取得了 2007 年 PASCAL VOC 物体检测竞赛的冠军。底层特征采用了Dalal 和 Triggs 提出的 HOG 特征，但与 Dalal 等人的全局刚体模板模型不同的是，形变部件模型由一个根模型和若干可形变部件组成。另一个重要的改进是提出了隐支持向量机模型，通过隐变量来建模物体部件的空间配置，并使用判别式方法进行训练优化。形变部件模型奠定了当今物体检测算法研究的基础，也成为后续 PASCAL VOC 竞赛物体检测任务的基础框架。
2008 年物体检测冠军同样采用了滑动窗口方式。特征表达利用了 HOG 特征和基于密集提取SIFT 的词包模型表达。训练过程对前、后、左、右分别训练独立的模型，并使用线性分类器和卡方核SVM 进行分类。测试过程采用了两阶段算法，第一阶段通过滑动窗口方式利用分类器得到大量可能出现物体的位置，第二阶段基于 HOG 和 SIFT 特征对前面一阶段得到的检测进行打分，最后使用非极大抑制算法去除错误检测窗口，并融合分类结果得到最终检测结果。这里分类信息可以看成是一种上下文信息，这个也是物体检测研究的一个重要内容。
2009 年除了形变部件模型以外，牛津大学视觉几何研究组在滑动窗口框架下，基于多核学习将灰度 PHOW、颜色 PHOW、PHOC、对称 PHOG、SSIM、视觉词典等多种特征进行融合，取得了与形变部件模型相近的效果，获得共同检测冠军。多核学习是进行多特征、多模型融合的重要策略，可以自动学习多个核矩阵的权重，从而得到最佳的模型融合效果。考虑到滑动窗口搜索的效率问题，提出了
类似级联 Adaboost 方式的多级分类器结构。第一级分类器采用线性 SVM 分类器以滑动窗口或者跳跃窗口方式快速对图像窗口进行粗分类；第二级采用拟线性 SVM,利用卡方核进行进一步细分类；第三级采用更强的非线性卡方-RBF 分类器，这一步准确度更高但比前面步骤计算代价更大，由于前面两级已经快速滤除大部分备选窗口，这一级可以专注于更难的样本分类。
2010 年中国科学院自动化研究所模式识别国家重点实验室获得了物体检测冠军，其方法是以形变部件模型为基础，对底层 HOG 特征进行了改进，提出了 Boosted HOG-LBP 特征，利用Gentle Boost 选择出一部分 LBP 特征与 HOG 特征融合，使得物体检测结果产生显著提升。另一个重要改进是采用了多种形状上下文，包括空间上下文、全局上下文、类间上下文。空间上下文由包含了窗口位置尺度信息的 6 维向量构成，全局上下文包括 20 维的物体分类分数和 20 维的最大窗口分数，其中分类方法采用了黄等人提出的显著性编码、词典关系算法计算词包模型表达。类间上下文用于建模相邻物体之间的弱空间关系，分别由20 维的窗口附近最强的 HOG 特征分数和 LBP 特征分数构成。最终得到 87 维的特征，使用 RBF SVM进行上下文学习。该方法在 VOC2010 数据库上取得了 6 项第一，5 项第二，平均精度达到了 36.8%。
2011 年物体检测冠军依然是中国科学院自动化研究所模式识别国家重点实验室，算法上与2010 年不同之处是针对形变部件模型提出了一种数据分解算法，并引入了空间混合建模和上下文学习。
2012 年阿姆斯特丹大学获得物体检测冠军，其方法主要创新在于选择性搜索、混合特征编码、新的颜色描述子、再训练过程。图像中物体本身构成一种层次结构，通常很难在一个尺度上检测所有物体，因而对图像块进行层次化组织，在每个层次上进行选择搜索，可以有效提升检测的召回率。考虑到经典的向量量化编码使用小的特征空间分块能够捕获更多图像细节，而丢失了分块内部的细节，而超向量编码和 Fisher 向量量化编码等差异编码方法则可以很好的描述分块内部细节，更大空间分块可以描述更大范围的图像细节，综合这两种编码模式，提出了混合特征编码算法，将两种编码的优点融合到一起。

**结论**

物体分类与检测在计算机视觉研究中具有重要的理论意义和实际应用价值，同时目前也存在诸多困难与挑战。本文以计算机视觉目标识别竞赛PASCAL VOC 为主线，对物体分类与检测历年最佳算法的发展进行了详尽的阐述，强调了表达学习和结构学习分别在物体分类和物体检测中的重要意义。以此为基础，本文还讨论了物体分类与检测的统一性与差异性，对物体分类与检测的发展方向进一步思考，从基于深度学习的表达学习和结构学习两个方向进行了分析与展望。


```python

```


```python

```


```python
#导入需要的包
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import paddle
print("本教程基于Paddle的版本号为："+paddle.__version__)
```

    本教程基于Paddle的版本号为：2.1.0


# （一）准备数据

(1)数据集介绍

MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。

![](https://ai-studio-static-online.cdn.bcebos.com/fc73217ae57f451a89badc801a903bb742e42eabd9434ecc8089efe19a66c076)

(2)transform函数是定义了一个归一化标准化的标准

(3)train_dataset和test_dataset

paddle.vision.datasets.MNIST()中的mode='train'和mode='test'分别用于获取mnist训练集和测试集

transform=transform参数则为归一化标准


```python
#导入数据集Compose的作用是将用于数据集预处理的接口以列表的方式进行组合。
#导入数据集Normalize的作用是图像归一化处理，支持两种方式： 1. 用统一的均值和标准差值对图像的每个通道进行归一化处理； 2. 对每个通道指定不同的均值和标准差值进行归一化处理。
from paddle.vision.transforms import Compose, Normalize, Resize, RandomRotation, RandomCrop
img_size = 32
#对训练集做数据增强
transform1 = Compose([Resize((img_size+2, img_size+2)), RandomCrop(img_size), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
transform2 = Compose([Resize((img_size, img_size)), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
# 使用transform对数据集做归一化
print('下载并加载训练数据')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform1)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform2)
print('加载完成')
```

    下载并加载训练数据
    加载完成


# （二）搭建网络
```
所搭建的网络不包括输入层的情况下，共有7层：5个卷积层、2个全连接层
其中第一个卷积层的输入通道数为数据集图片的实际通道数。MNIST数据集为灰度图像，通道数为1
第1个卷积层输出与第3个卷积层输出做残差作为第4个卷积层的输入，第4个卷积层的输入与第5个卷积层的输出做残差作为第1个全连接层的输入
```


```python
import paddle
import paddle.nn.functional as F
# 定义多层卷积神经网络
#动态图定义多层卷积神经网络
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = paddle.nn.Conv2D(6, 16, 3, padding=1)
        self.conv3 = paddle.nn.Conv2D(16, 32, 3, padding=1)
        self.conv4 = paddle.nn.Conv2D(6, 32, 1)
        self.conv5 = paddle.nn.Conv2D(32, 64, 3, padding=1)
        self.conv6 = paddle.nn.Conv2D(64, 128, 3, padding=1)
        self.conv7 = paddle.nn.Conv2D(32, 128, 1)
        self.maxpool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.maxpool2 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool3 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool4 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool5 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool6 = paddle.nn.MaxPool2D(2, 2)
        self.flatten=paddle.nn.Flatten()
        self.linear1=paddle.nn.Linear(128, 128)
        self.linear2=paddle.nn.Linear(128, 10)
        self.dropout=paddle.nn.Dropout(0.2)
        self.avgpool=paddle.nn.AdaptiveAvgPool2D(output_size=1)
        
    def forward(self, x):
        y = self.conv1(x)#(bs 6, 32, 32)
        y = F.relu(y)
        y = self.maxpool1(y)#(bs, 6, 16, 16)
        z = y
        y = self.conv2(y)#(bs, 16, 16, 16)
        y = F.relu(y)
        y = self.maxpool2(y)#(bs, 16, 8, 8)
        y = self.conv3(y)#(bs, 32, 8, 8)
        z = self.maxpool4(self.conv4(z))
        y = y+z
        y = F.relu(y)
        z = y
        y = self.conv5(y)#(bs, 64, 8, 8)
        y = F.relu(y)
        y = self.maxpool5(y)#(bs, 64, 4, 4)
        y = self.conv6(y)#(bs, 128, 4, 4)
        z = self.maxpool6(self.conv7(z))
        y = y + z
        y = F.relu(y)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return y
```


```python
import os, sys
from PIL import Image
import numpy as np
import pandas as pd
import paddle
from paddle import fluid
from paddle.fluid.layers import data, conv2d, pool2d, flatten, fc, cross_entropy, accuracy, mean, concat, dropout


class InceptionV1:
    def __init__(self, structShow=False):
        self.structShow = structShow
        self.image = data(shape=[Img_chs, Img_size, Img_size], dtype='float32', name='image')
        self.label = data(shape=[Label_size], dtype='int64', name='label')
        self.predict = self.get_Net()
 
    def InceptionV1_Model(self, input, model_size):
        con11_chs, con31_chs, con3_chs, con51_chs, con5_chs, pool1_chs = model_size
 
        conv11 = conv2d(input, con11_chs, filter_size=1, padding='SAME', act='relu')
 
        conv31 = conv2d(input, con31_chs, filter_size=1, padding='SAME', act='relu')
        conv3 = conv2d(conv31, con3_chs, filter_size=3, padding='SAME', act='relu')
 
        conv51 = conv2d(input, con51_chs, filter_size=1, padding='SAME', act='relu')
        conv5 = conv2d(conv51, con5_chs, filter_size=5, padding='SAME', act='relu')
 
        pool1 = pool2d(input, pool_size=3, pool_stride=1, pool_type='max', pool_padding='SAME')
        conv1 = conv2d(pool1, pool1_chs, filter_size=1, padding='SAME', act='relu')
 
        output = concat([conv11, conv3, conv5, conv1], axis=1)
        return output
 
    def InceptionV1_Out(self, input, name=None):
        pool = pool2d(input, pool_size=5, pool_stride=3, pool_type='avg', pool_padding='VALID')
 
        conv = conv2d(pool, Out_chs1, filter_size=1, padding='SAME', act='relu')
 
        flat = flatten(conv, axis=1)
        dp = dropout(flat, 0.3)
        output = fc(dp, Labels_nums, name=name)
        return output
 
    def get_Net(self):
        # region conv pool
        conv1 = conv2d(self.image, Conv1_chs, filter_size=Conv1_kernel_size, stride=2, padding='SAME', act='relu')
        pool1 = pool2d(conv1, pool_size=3, pool_stride=2, pool_type='max', pool_padding='SAME')
 
        conv21 = conv2d(pool1, Conv21_chs, filter_size=Conv21_kernel_size, padding='SAME', act='relu')
        conv2 = conv2d(conv21, Conv2_chs, filter_size=Conv2_kernel_size, padding='SAME', act='relu')
        pool2 = pool2d(conv2, pool_size=3, pool_stride=2, pool_type='max', pool_padding='SAME')
        # endregion
 
        # region inception3
        inception3a = self.InceptionV1_Model(pool2, Icp3a_size)
 
        inception3b = self.InceptionV1_Model(inception3a, Icp3b_size)
        pool3 = pool2d(inception3b, pool_size=3, pool_stride=2, pool_type='max', pool_padding='SAME')
        # endregion
 
        # region inception3
        inception4a = self.InceptionV1_Model(pool3, Icp4a_size)
        output1 = self.InceptionV1_Out(inception4a, 'output1')
 
        inception4b = self.InceptionV1_Model(inception4a, Icp4b_size)
 
        inception4c = self.InceptionV1_Model(inception4b, Icp4c_size)
 
        inception4d = self.InceptionV1_Model(inception4c, Icp4d_size)
        output2 = self.InceptionV1_Out(inception4d, 'output2')
 
        inception4e = self.InceptionV1_Model(inception4d, Icp4e_size)
        pool4 = pool2d(inception4e, pool_size=3, pool_stride=2, pool_type='max', pool_padding='SAME')
        # endregion
 
        # region inception5
        inception5a = self.InceptionV1_Model(pool4, Icp5a_size)
 
        inception5b = self.InceptionV1_Model(inception5a, Icp5b_size)
        pool5 = pool2d(inception5b, pool_size=7, pool_stride=1, pool_type='max', pool_padding='SAME')
        # endregion
 
        # region output
        flat = flatten(pool5, axis=1)
        dp = dropout(flat, 0.4)
        output = fc(dp, Labels_nums, name='output')
        # endregion
 
        if self.structShow:
            print(pool1.name, pool1.shape)
            print(pool2.name, pool2.shape)
 
            print(inception3a.name, inception3a.shape)
            print(inception3b.name, inception3b.shape)
            print(pool3.name, pool3.shape)
 
            print(inception4a.name, inception4a.shape)
            print(output1.name, output1.shape)
            print(inception4b.name, inception4b.shape)
            print(inception4c.name, inception4c.shape)
            print(inception4d.name, inception4d.shape)
            print(output2.name, output2.shape)
            print(inception4e.name, inception4e.shape)
            print(pool4.name, pool4.shape)
 
            print(inception5a.name, inception5a.shape)
            print(inception5b.name, inception5b.shape)
            print(pool5.name, pool5.shape)
 
            print(flat.name, flat.shape)
            print(output.name, output.shape)
            print(output.name, output.shape)
 
        return [output, output1, output2]


```


```python
#定义卷积网络的代码
net_cls = InceptionV1()
paddle.summary(net_cls, (-1, 1, img_size, img_size))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-25-4c3bd4fbe6bb> in <module>
          1 #定义卷积网络的代码
    ----> 2 net_cls = InceptionV1()
          3 paddle.summary(net_cls, (1, img_size, img_size))


    <ipython-input-23-77ae22691d3c> in __init__(self, structShow)
         11     def __init__(self, structShow=False):
         12         self.structShow = structShow
    ---> 13         self.image = data(shape=[Img_chs, Img_size, Img_size], dtype='float32', name='image')
         14         self.label = data(shape=[Label_size], dtype='int64', name='label')
         15         self.predict = self.get_Net()


    NameError: name 'Img_chs' is not defined


# （三）参数设置及模型训练


```python
from paddle.metric import Accuracy
save_dir = "output/model/v5_7"
patience = 5
epoch = 50
lr = 0.01
weight_decay = 5e-4
batch_size = 64
momentum = 0.9
# 用Model封装模型
model = paddle.Model(net_cls)

# 定义损失函数
#optim = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=weight_decay)
#lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=10000, eta_min=1e-5)
optim = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters(), momentum=momentum)

visual_dl = paddle.callbacks.VisualDL(log_dir=save_dir)
early_stop = paddle.callbacks.EarlyStopping(monitor='acc', mode='max', patience=patience, 
                                            verbose=1, min_delta=0, baseline=None,
                                            save_best_model=True)
# 配置模型
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())

# 训练保存并验证模型
model.fit(train_dataset,test_dataset,epochs=epoch,batch_size=batch_size,
        save_dir=save_dir,verbose=1, callbacks=[visual_dl, early_stop])
```

# （四）模型验证及测试


```python
best_model_path = "work/best_model.pdparams"
net_cls = MyNet()
model = paddle.Model(net_cls)
model.load(best_model_path)
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())
```


```python
#用最好的模型在测试集10000张图片上验证
results = model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
print(results)
```


```python
########测试
#获取测试集的第一个图片
test_data0, test_label_0 = test_dataset[0][0],test_dataset[0][1]
test_data0 = test_data0.reshape([img_size,img_size])
plt.figure(figsize=(2,2))
#展示测试集中的第一个图片
print(plt.imshow(test_data0, cmap=plt.cm.binary))
print('test_data0 的标签为: ' + str(test_label_0))
#模型预测
result = model.predict(test_dataset, batch_size=1)
#打印模型预测的结果
print('test_data0 预测的数值为：%d' % np.argsort(result[0][0])[0][-1])
```


```python

```


```python

```
