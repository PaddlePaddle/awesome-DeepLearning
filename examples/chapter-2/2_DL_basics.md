#  损失函数方法补充

softmax loss实际上是由softmax和cross-entropy loss组合而成，两者放一起数值计算更加稳定。

令z是softmax层的输入，f(z)是softmax的输出，则

![图片](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONynNhvrnE9z9MN6qshkEP9vNIickdaYDSFekBLnX8A7Zu1z2nsVKfBnTDymjiaiaV8TzbDoTLkbHUSjicQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

单个像素i的softmax loss等于cross-entropy error如下:

![图片](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONynNhvrnE9z9MN6qshkEP9vNds3jKoqNRfkNHR3JYlia6nwW8ZDgQib7TIGWlgdWSuh71BOFozM7q6LQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

展开上式：

![图片](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONynNhvrnE9z9MN6qshkEP9vNhaVF5XdfEkNnOHCx1vicumDVwmn7nFEKiawqXfcPxiaNrxCFYt9e3piahQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在caffe实现中，z即bottom blob，l(y,z)是top blob，反向传播时，就是要根据top blob diff得到bottom blob diff,所以要得到 

![图片](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONynNhvrnE9z9MN6qshkEP9vNZVLbMep1skIX3j5n6ncx6Iz20hH6OhoBn3uFupdvoHrKeXobmqSZUA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

下面求loss对z的第k个节点的梯度

![图片](https://mmbiz.qpic.cn/mmbiz_png/AmjGbfdONynNhvrnE9z9MN6qshkEP9vNAfJibu35znypibLibibB4r9ZDqibUxVVatibJRicVJiasCic9M1hRDCuxEQlOicA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可见，传给groundtruth label节点和非groundtruth label节点的梯度是不一样的。

#  损失函数Python代码实现

>#conding=utf-8
>from torch import nn
>import torch
>import numpy as np
>def MySoftmax(vector):
>return np.exp(vector)/np.exp(vector).sum()
>def LossFunc(target,output):
>output = MySoftmax(output)
>one_hot = np.zeros_like(output)
>one_hot[:,target] = 1
>#print(one_hot)
>loss = (-np.log(output)*one_hot).sum()
>return loss
>target = np.array([1])
>output = np.array([[8,-3.,10]])
>softmax_out = MySoftmax(output)
>np.set_printoptions(suppress=True)
>print(softmax_out)
>#torch自带的softmax实现
>print(nn.Softmax()(torch.Tensor(output)))
>print(LossFunc(target,output))
>print(nn.CrossEntropyLoss(reduction="sum")(torch.Tensor(output),torch.Tensor(target).long()))

#  池化方法补充

**池化的作用则体现在降采样：保留显著特征、降低特征维度，增大kernel的感受野**。深度网络越往后面越能捕捉到物体的语义信息，这种语义信息是建立在较大的感受野基础上。

 

### 1、一般池化(General Pooling)

<img src="https://www.pianshen.com/images/30/bb4dd12d2c1247f5c18e063efd3fdbce.gif" alt="img" style="zoom:80%;" />

池化作用于图像中不重合的区域（与卷积操作不同），定义池化窗口的大小为sizeX，即图中红色正方形的边长，定义两个相邻池化窗口的水平位移 / 竖直位移为stride。一般池化由于每一池化窗口都是不重复的，所以**sizeX=stride**。

[池化方法总结（Pooling）](https://blog.csdn.net/danieljianfeng/article/details/42433475)

 

### 2、均值池化(Mean / Average Pooling)

<img src="https://www.pianshen.com/images/15/7700c08ad172406406f2fa52a224d3af.png" alt="img" style="zoom:67%;" />

一般池化的基础上，计算每个池化窗口对应图像区域的**平均值**，作为该区域池化后的值。

 

### 3、最大池化(Max Pooling)

<img src="https://www.pianshen.com/images/58/9da3612d1ca10124c401819c6c3af8ea.png" alt="img" style="zoom: 33%;" />

一般池化的基础上，选取每个池化窗口对应图像区域的**最大值**，作为该区域池化后的值。

 

### 4、随机池化(Stochastic Pooling)

<img src="https://www.pianshen.com/images/219/0d809e4b3b01f89ef683b21e5644d723.png" alt="img" style="zoom: 50%;" />

Stochastic pooling是**一种简单有效的正则化CNN的方法，能够降低max pooling的过拟合现象，提高泛化能力**。对于pooling层的输入，根据输入的多项式分布随机选择一个值作为输出。训练阶段和测试阶段的操作略有不同。

**训练阶段：**

1）前向传播：先将池化窗口中的元素全部除以它们的和，得到概率矩阵；再按照概率随机选中的方格的值，作为该区域池化后的值。

2）反向传播：求导时，只需保留前向传播中已经被选中节点的位置的值，其它值都为0，类似max-pooling的反向传播。

**测试阶段：**

在测试时也使用Stochastic Pooling会对预测值引入噪音，降低性能。取而代之的是使用概率矩阵加权平均。比使用Average Pooling表现要好一些。在平均意义上，与Average Pooling近似，在局部意义上，服从Max Pooling准则。

### 5、重叠池化(Overlapping Pooling)

重叠池化，即相邻池化窗口之间会有重叠区域。如果定义池化窗口的大小为sizeX，定义两个相邻池化窗口的水平位移 / 竖直位移为stride，此时**sizeX>stride**。

Alexnet中提出和使用，不仅可以提升预测精度，同时一定程度上可以减缓过拟合。相比于正常池化（步长s=2，窗口x=2），重叠池化(步长s=2，窗口x=3) 可以减少top-1, top-5的错误率分别为0.4% 和0.3%。

### 6、全局池化(Global Pooling)

Global Pooling就是**池化窗口的大小 = 整张特征图的大小**。这样，每个 W×H×C 的特征图输入就会被转化为 1×1×C 的输出，也等同于每个位置权重都为 1/(W×H) 的全连接层操作。


#  数据增强方法补充

有监督数据增强，即**采用预设的数据变换规则，在已有数据的基础上**进行数据的扩增，包含单样本数据增强和多样本数据增强，其中单样本又包括几何操作类，颜色变换类。

**1. 单样本数据增强**

所谓单样本数据增强，即增强一个样本的时候，全部围绕着该样本本身进行操作，包括**几何变换类，颜色变换类**等。

**(1) 几何变换类**

几何变换类即对图像进行几何变换，包括**翻转，旋转，裁剪，变形，缩放**等各类操作。

**(2) 颜色变换类**

上面的几何变换类操作，没有改变图像本身的内容，它可能是选择了图像的一部分或者对像素进行了重分布。如果要改变图像本身的内容，就属于颜色变换类的数据增强了，常见的包括**噪声、模糊、颜色变换、擦除、填充**等等。

基于噪声的数据增强就是在原来的图片的基础上，随机叠加一些噪声，最常见的做法就是高斯噪声。更复杂一点的就是在面积大小可选定、位置随机的矩形区域上丢弃像素产生黑色矩形块，从而产生一些彩色噪声，以Coarse Dropout方法为代表，甚至还可以对图片上随机选取一块区域并擦除图像信息。

颜色变换的另一个重要变换是颜色扰动，就是在某一个颜色空间通过增加或减少某些颜色分量，或者更改颜色通道的顺序。

**2. 多样本数据增强**

不同于单样本数据增强，多样本数据增强方法利用多个样本来产生新的样本，下面介绍几种方法。

**(1) SMOTE**

SMOTE即Synthetic Minority Over-sampling Technique方法，它是通过人工合成新样本来处理样本不平衡问题，从而提升分类器性能。

类不平衡现象是很常见的，它指的是数据集中各类别数量不近似相等。如果样本类别之间相差很大，会影响分类器的分类效果。假设小样本数据数量极少，如仅占总体的1%，则即使小样本被错误地全部识别为大样本，在经验风险最小化策略下的分类器识别准确率仍能达到99%，但由于没有学习到小样本的特征，实际分类效果就会很差。

SMOTE方法是基于插值的方法，它可以为小样本类合成新的样本。

**(2) SamplePairing**

SamplePairing方法的原理非常简单，从训练集中随机抽取两张图片分别经过基础数据增强操作(如随机翻转等)处理后经像素以取平均值的形式叠加合成一个新的样本，标签为原样本标签中的一种。这两张图片甚至不限制为同一类别，这种方法对于医学图像比较有效。

经SamplePairing处理后可使训练集的规模从N扩增到N×N。实验结果表明，因SamplePairing数据增强操作可能引入不同标签的训练样本，导致在各数据集上使用SamplePairing训练的误差明显增加，而在验证集上误差则有较大幅度降低。

尽管SamplePairing思路简单，性能上提升效果可观，符合奥卡姆剃刀原理，但遗憾的是可解释性不强。

**(3) mixup**

mixup是Facebook人工智能研究院和MIT在“Beyond Empirical Risk Minimization”中提出的基于邻域风险最小化原则的数据增强方法，它使用线性插值得到新样本数据。

**SMOTE，SamplePairing，mixup三者思路上有相同之处，都是试图将离散样本点连续化来拟合真实样本分布**，不过所增加的样本点在特征空间中仍位于已知小样本点所围成的区域内。如果能够在给定范围之外适当插值，也许能实现更好的数据增强效果。

**3. 无监督的数据增强**

无监督的数据增强方法包括两类：

(1) 通过模型学习数据的分布，随机生成与训练数据集分布一致的图片，代表方法GAN[4]。

(2) 通过模型，学习出适合当前任务的数据增强方法，代表方法AutoAugment[5]。

**3.1 GAN**

关于GAN(generative adversarial networks)，我们已经说的太多了。它包含两个网络，一个是生成网络，一个是对抗网络，基本原理如下：

(1) G是一个生成图片的网络，它接收随机的噪声z，通过噪声生成图片，记做G(z) 。

(2) D是一个判别网络，判别一张图片是不是“真实的”，即是真实的图片，还是由G生成的图片。

**3.2 Autoaugmentation[5]**

AutoAugment是Google提出的自动选择最优数据增强方案的研究，这是无监督数据增强的重要研究方向。它的基本思路是使用增强学习从数据本身寻找最佳图像变换策略，对于不同的任务学习不同的增强方法，流程如下：

(1) 准备16个常用的数据增强操作。

(2) 从16个中选择5个操作，随机产生使用该操作的概率和相应的幅度，将其称为一个sub-policy，一共产生5个sub-polices。

(3) 对训练过程中每一个batch的图片，随机采用5个sub-polices操作中的一种。

(4) 通过模型在验证集上的泛化能力来反馈，使用的优化方法是增强学习方法。

(5) 经过80~100个epoch后网络开始学习到有效的sub-policies。

(6) 之后串接这5个sub-policies，然后再进行最后的训练。

总的来说，就是学习已有数据增强的组合策略，对于门牌数字识别等任务，研究表明剪切和平移等几何变换能够获得最佳效果。

#  图像分类方法综述

**基于色彩特征的索引技术**

色彩是物体表面的一种视觉特性,每种物体都有其特有的色彩特征,譬如人们说到绿色往往是和树木或草原相关,谈到蓝色往往是和大海或蓝天相关,同一类物体往拍几有着相似的色彩特征,因此我们可以根据色彩特征来区分物体.用色彩特特征进行图像分类一可以追溯到Swain和Ballard提出的色彩直方图的方法.由于色彩直方图具有简单且随图像的大小、旋转变化不敏感等特点,得到了研究人员的厂泛关注,目前几乎所有基于内容分类的图像数据库系统都把色彩分类方法作为分类的一个重要手段,并提出了许多改进方法,归纳起主要可以分为两类：全局色彩特征索引和局部色彩特征索引。

**基于纹理的图像分类技术**

纹理特征也是图像的重要特征之一,其本质是刻画象素的邻域灰度空间分布规律由于它在模式识别和计算机视觉等领域已经取得了丰富的研究成果,因此可以借用到图像分类中。

在70年代早期,Haralick等人提出纹理特征的灰度共生矩阵表示法(eo一oeeurrenee matrix representation),这个方法提取的是纹理的灰度级空间相关性(gray level Spatial dependenee),它首先基于象素之间的距离和方向建立灰度共生矩阵,再由这个矩阵提取有意义的统计量作为纹理特征向量。基于一项人眼对纹理的视觉感知的心理研究,Tamuar等人提出可以模拟纹理视觉模型的6个纹理属性,分别是粒度,对比度,方向性,线型,均匀性和粗糙度。QBIC系统和MARS系统就采用的是这种纹理表示方法。

在90年代初期,当小波变换的理论结构建一认起来之后,许多研究者开始研究

如何用小波变换表示纹理特征。smiht和chang利用从小波子带中提取的统计量(平均值和方差)作为纹理特征。这个算法在112幅Brodatz纹理图像中达到了90%的准确率。为了利用中间带的特征,Chang和Kuo开发出一种树型结构的小波变化来进一步提高分类的准确性。还有一些研究者将小波变换和其他的变换结合起来以得到更好的性能,如Thygaarajna等人结合小波变换和共生矩阵,以兼顾基于统计的和基于变换的纹理分析算法的优点。

**基于形状的图像分类技术**

形状是图像的重要可视化内容之一在二维图像空间中,形状通常被认为是一条封闭的轮廓曲线所包围的区域,所以对形状的描述涉及到对轮廓边界的描述以及对这个边界所包围区域的描述.目前的基于形状分类方法大多围绕着从形状的轮廓特征和形状的区域特征建立图像索引。关于对形状轮廓特征的描述主要有:直线段描述、样条拟合曲线、傅立叶描述子以及高斯参数曲线等等。

实际上更常用的办法是采用区域特征和边界特征相结合来进行形状的相似分类.如Eakins等人提出了一组重画规则并对形状轮廓用线段和圆弧进行简化表达,然后定义形状的邻接族和形族两种分族函数对形状进行分类.邻接分族主要采用了形状的边界信息,而形状形族主要采用了形状区域信息.在形状进行匹配时,除了每个族中形状差异外,还比较每个族中质心和周长的差异,以及整个形状的位置特征矢量的差异,查询判别距离是这些差异的加权和。

**基于空间关系的图像分类技术**

在图像信息系统中,依据图像中对象及对象间的空间位置关系来区别图像库中的不同图像是一个非常重要的方法。因此,如何存贮图像对象及其中对象位置关系以方便图像的分类,是图像数据库系统设计的一个重要问题。而且利用图像中对象间的空间关系来区别图像,符合人们识别图像的习惯,所以许多研究人员从图像中对象空间位置关系出发,着手对基于对象空间位置关系的分类方法进行了研究。早在1976年,Tanimoto提出了用像元方法来表示图像中的实体,并提出了用像元来作为图像对象索引。随后被美国匹兹堡大学chang采纳并提出用二维符号串(2D一String)的表示方法来进行图像空间关系的分类,由于该方法简单,并且对于部分图像来说可以从ZD一String重构它们的符号图,因此被许多人采用和改进,该方法的缺点是仅用对象的质心表示空间位置;其次是对于一些图像来

说我们不能根据其ZD一string完个重构其符号图;再则是上述的空间关系太简单,实际中的空间关系要复杂得多。,针对这些问题许多人提出了改进力一法。Jungert根据图像对象的最小包围盒分别在:x轴方向和y轴上的投影区间之间的交叠关系来表示对象之间的空间关系,随后Cllallg和Jungert等人又提出了广义ZD一string(ZDG一String)的方法,将图像对象进一步切分为更小的子对象来表示对象的空间关系,但是该方法不足之处是当图像对象数日比较多且空间关系比较复杂时,需要切分的子对象的数目很多,存储的开销太大,针对此Lee和Hsu等人提出了ZDC一string的方一法,它们采用Anell提出的13种时态间隔关系并应用到空间投影区问上来表达空间关系。在x轴方向和y轴方向的组合关系共有169种,他提出了5种基本关系转换法则,在此基础上又提出了新的对象切分方法。采用

ZDC一string的方法比ZDG一string切分子对象的数目明显减少。为了在空间关系中保留两个对象的相对空间距离和对象的大小,Huang等人提出了ZDC书string的方法提高符号图的重构精度,并使得对包含对象相对大小、距离的符号图的推理成为可能。上述方法都涉及到将图像对象进行划分为子对象,且在用符号串重构对象时处理时间的开销都比较大,为解决这些方法的不足,Lee等人又提出了ZDB一String的方法,它不要求对象进一步划分,用对象的名称来表示对象的起点和终点边界。为了解决符号图的重构问题,Chin一ChenCllang等人提出了面向相对坐标解决符号图的重构问题,Chin一ChenChang等人提出了面向相对坐标符号串表示(RCOS串),它们用对象最小外接包围盒的左下角坐标和右上角坐标来表示对象之间的空间关系.

对于对象之间的空间关系采用Allen提出的13种区间表示方法。实际上上述所有方法都不是和对象的方位无关,为此Huang等人又提出了RSString表示方法。虽然上述各种方法在对图像对象空间信息的分类起到过一定作用,由于它们都是采用对象的最小外接矩形来表示一个对象空间位置,这对于矩形对象来说是比较合适的,但是当两个对象是不规则形状,且它们在空间关系上是分离时,它们的外接矩形却存在着某种包含和交叠,结果出现对这些对象空间关系的错误表示。用上述空间关系进行图像分类都是定性的分类方一法,将图像的空间关系转换为图像相似性的定量度量是一个较为困难的事情。Nabil综合ZD一String方法和二维平面中对象之间的点集拓扑关系。提出了ZD一PIR分类方法,两个对象之间的相似与否就转换为两个图像的ZD一PIR图之间是否同构。ZD一PIR中只有图像对象之间的空间拓扑关系具有旋转不变性,在进行图像分类的时候没有考虑对象之间的相对距离。
