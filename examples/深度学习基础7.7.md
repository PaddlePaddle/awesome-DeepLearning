## 深度学习基础知识7.7

### 1.损失函数补充
损失函数（loss function）是用来估量模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。模型的结构风险函数包括了经验风险项和正则项，通常可以表示成如下式子：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/491809382acb44f1b247e14c3da7bcc36f4c4c5ac0664cb8952486d8a264a4bf" width="500" hegiht="" ></center>
<br></br>

其中，前面的均值函数表示的是经验风险函数，L代表的是损失函数，后面的Φ是正则化项（regularizer）或者叫惩罚项（penalty term），它可以是L1，也可以是L2，或者其他的正则函数。整个式子表示的意思是找到使目标函数最小时的θ值。下面主要列出几种常见的损失函数。

理解：损失函数旨在表示出logit和label的差异程度，不同的损失函数有不同的表示意义，也就是在最小化损失函数过程中，logit逼近label的方式不同，得到的结果可能也不同。

一般情况下，softmax和sigmoid使用交叉熵损失（logloss），hingeloss是SVM推导出的，hingeloss的输入使用原始logit即可。

#### 1.1交叉熵损失的拓展——Focal Loss

我们已经对交叉熵函数很熟悉了，它的公式为：
$$H\left( p,q \right) =-\sum_x{p\left( x \right) \log q\left( x \right)}$$
其中 $p$ 和 $q$ 是数据 $x$ 的两个概率分布。

二分类交叉熵损失为：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/a243666ed53c4f2095a3f93ef7bf54def254d3b21a5f4cbfb214790bbe5cb029" width="500" hegiht="" ></center>
<br></br>

$y'$是经过激活函数的输出，所以在0-1之间。可见普通的交叉熵对于正样本而言，输出概率越大损失越小。对于负样本而言，输出概率越小则损失越小。此时的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。

Focal loss主要是为了解决one-stage目标检测中正负样本比例严重失衡的问题,它在交叉熵损失函数基础上进行的修改：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/4369c78cc6c14270b5d19893b86255cc23b4c9de587140e5b5f66544e4d70092" width="500" hegiht="" ></center>
<br></br>


首先在原有的基础上加了一个因子，其中gamma>0使得减少易分类样本的损失。使得更关注于困难的、错分的样本。此外，加入平衡因子alpha，用来平衡正负样本本身的比例不均。

```
import numpy as np
def Focalloss(Y, P，gamma,alpha):
    """
    Y 是经过激活函数的输出
    P 是标签
    """
    Y = np.float_(Y)
    P = np.float_(P)
    return -alpha*pow((1-Y),gamma)*log(Y)*P-(1-alpha)*pow（Y,gamma）*log(1-Y)*(1-P)
```

#### 1.2指数损失函数（Adaboost）
Adaboost是一种集成算法，假设训练样本集为：
$$T=\{(x_,y_1),(x_2,y_2), ...(x_m,y_m)\}$$
训练集的在第k个弱学习器的输出权重为:
$$D(k) = (w_{k1}, w_{k2}, ...w_{km}) ;\;\; w_{1i}=\frac{1}{m};\;\; i =1,2...m$$
Adaboost是模型为加法模型，学习算法为前向分步学习算法，损失函数为指数函数的分类问题。
Adaboost是一个利用前一个强学习器的结果和当前弱学习器来更新当前的强学习器的模型，第k-1轮的强学习器为：
$$f_{k-1}(x) = \sum\limits_{i=1}^{k-1}\alpha_iG_{i}(x)$$
而第k轮的强学习器为:
$$f_{k}(x) = \sum\limits_{i=1}^{k}\alpha_iG_{i}(x)$$
从而我们可以得到：
$$f_{k}(x) = f_{k-1}(x) + \alpha_kG_k(x)$$
Adaboost损失函数为指数函数，即定义损失函数为:
$$\underbrace{arg\;min\;}_{\alpha, G} \sum\limits_{i=1}^{m}exp(-y_if_{k}(x))$$
　利用前向分步学习算法的关系可以得到损失函数为:
 $$(\alpha_k, G_k(x)) = \underbrace{arg\;min\;}_{\alpha, G}\sum\limits_{i=1}^{m}exp[(-y_i) (f_{k-1}(x) + \alpha G(x))]$$
 
```
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
def adaboostloss(Y_train, X_train, Y_test, X_test):
    n_train, n_test = len(X_train), len(X_test)
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f"
         % (i + 1, 1 - sum(miss) / n_train))

        err_m = np.dot(w, miss)

        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        miss2 = [x if x==1 else -1 for x in miss] # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)
        
     return w,miss2

```

#### 1.3Hinge损失函数（SVM）
在机器学习算法中，hinge损失函数和SVM是息息相关的。在线性支持向量机中，最优化问题可以等价于下列式子：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8854204f469c48648a5564048d69d1a50e4a26764fb440708cbf8bd164b4a257" width="500" hegiht="" ></center>

如果令：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/2290160105504b658b4f5af255690edef6705ef8dd464baaa50134d97e4a5783" width="300" hegiht="" ></center>

于是，原式就变成了：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/6be4101719ab49168be1fe5dfcf2a5dd54e8a3dd603049f480bb9c910890a6cb" width="300" hegiht="" ></center>

如若取λ=1/(2C)，式子就可以表示成：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e3880babf2884c5baebc06a2104a7f8b4141b1035ad84a4ea63f6789b57fc205" width="300" hegiht="" ></center>

前半部分中的$l$就是hinge损失函数，而后面相当于L2正则项。

**Hinge 损失函数的标准形式**
$$L(y)=max(0,1-y\tilde{y}),y =\pm 1$$
```
from sklearn import svm
from sklearn.metrics import hinge_loss
X=[[0],[1]]
y=[-1,1] 
est=svm.LinearSVC(random_state=0) 
print(est.fit(X,y)) 
pred_decision=est.decision_function([[-2],[3],[0.5]]) 
print(pred_decision) 
print(hinge_loss([-1,1,1],pred_decision)) #结果
```






#### 1.4其它损失函数
除了以上这几种损失函数，常用的还有：
**0-1损失函数**
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/57952013aab946c6bf778f0f0f9e01c8f2c600f495214ebebf5b294881e59120" width="400" hegiht="" ></center>

```
def 0-1_loss(Y,P):
	if abs(P-Y)<T:
		return 0
	else:
		return 1
```
**绝对值损失函数**
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/4cd5931111e544b5a9bbf551b9849ed2cd1ffd0bde3e45d983e2294b369ab1fa" width="300" hegiht="" ></center>

```
def abs_error(P,Y):
    return abs（P-Y）
```

### 2.损失函数python实现
见每个损失函数下

### 3.池化方法补充
#### 3.1 一般池化（General Pooling）
池化作用于图像中不重合的区域，比较常见的有平均池化、最大池化、K-max池化，因为在awesome-DeepLearning repo中有过介绍。

**随机池化(Stochastic Pooling)**
随机池化的计算过程如下：
* 先将方格中的元素同时除以它们的和sum，得到概率矩阵。
* 按照概率随机选中方格。
* pooling得到的值就是方格位置的值。
随机池化只需对特征图中的元素按照其概率值大小随机选择，即元素值大的被选中的概率也大，而不像max-pooling那样，永远只取那个最大值元素，这使得随机池化具有更强的泛化能力。


#### 3.2 重叠池化（OverlappingPooling）
重叠池化正如其名字所说的，相邻池化窗口之间会有重叠区域，此时size>stride，相对于传统的no-overlapping pooling，采用Overlapping Pooling不仅可以提升预测精度，同时一定程度上可以减缓过拟合。

#### 3.3 空金字塔池化（Spatial Pyramid Pooling）
在一般的CNN结构中，在卷积层后面通常连接着全连接。而全连接层的特征数是固定的，所以在网络输入的时候，会固定输入的大小(fixed-size)。但在现实中，我们的输入的图像尺寸总是不能满足输入时要求的大小。然而通常的手法就是裁剪(crop)和拉伸(warp)。

这样做总是不好的：图像的纵横比(ratio aspect) 和 输入图像的尺寸是被改变的。这样就会扭曲原始的图像。而Kaiming He在这里提出了一个SPP(Spatial Pyramid Pooling)层能很好的解决这样的问题， SPP达到的效果不管输入的图片是什么尺度，都能够正确的传入网络.，SPP通常连接在最后一层卷积层。

具体实现方案如图：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/dc0553e0930e4550aa2fb75c0c0a35b8d14dade59fa946a9862107e840470464" width="600" hegiht="" ></center>


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/1e027f488255432ba27226c2123ad7e3ce918f2c2b9640fb8b4b8ae17a09cc06" width="500" hegiht="" ></center>
<center><br>CNN一般结构和SPP结构</br></center>
<br></br>

### 4.数据增强方法补充

数据增强也叫数据扩增，意思是在不实质性的增加数据的情况下，让有限的数据产生等价于更多数据的价值。数据增强可以分为，有监督的数据增强和无监督的数据增强方法。其中有监督的数据增强又可以分为单样本数据增强和多样本数据增强方法，无监督的数据增强分为生成新的数据和学习增强策略两个方向。

#### 4.1有监督的数据增强

有监督数据增强，即采用预设的数据变换规则，在已有数据的基础上进行数据的扩增，包含单样本数据增强和多样本数据增强，其中单样本又包括几何操作类，颜色变换类。

**几何变换类**

几何变换类即对图像进行几何变换，包括翻转，旋转，裁剪，变形，缩放等各类操作，下面展示其中的若干个操作。

**颜色变换类**
基于噪声的数据增强就是在原来的图片的基础上，随机叠加一些噪声，最常见的做法就是高斯噪声。更复杂一点的就是在面积大小可选定、位置随机的矩形区域上丢弃像素产生黑色矩形块，从而产生一些彩色噪声，以Coarse Dropout方法为代表，甚至还可以对图片上随机选取一块区域并擦除图像信息。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/9c90015f6e7b47aba96b83edba0879ca83a096bd8f3248d48992e5815e75fa7e" width="600" hegiht="" ></center>

**SMOTE**
SMOTE即Synthetic Minority Over-sampling Technique方法，它是通过人工合成新样本来处理样本不平衡问题，从而提升分类器性能。
SMOTE方法是基于插值的方法，它可以为小样本类合成新的样本，主要流程为：
第一步，定义好特征空间，将每个样本对应到特征空间中的某一点，根据样本不平衡比例确定好一个采样倍率N；
第二步，对每一个小样本类样本(x,y)，按欧氏距离找出K个最近邻样本，从中随机选取一个样本点，假设选择的近邻点为(xn,yn)。在特征空间中样本点与最近邻样本点的连线段上随机选取一点作为新样本点，满足以下公式：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/491656c1ac1e425eb32a7a2b3fa1884bafec94eb71304903a73e54f9e481ad4b" width="600" hegiht="" ></center>
第三步，重复以上的步骤，直到大、小样本数量平衡。

**SamplePairing**
SamplePairing方法是从训练集中随机抽取两张图片分别经过基础数据增强操作(如随机翻转等)处理后经像素以取平均值的形式叠加合成一个新的样本，标签为原样本标签中的一种。这两张图片甚至不限制为同一类别，这种方法对于医学图像比较有效。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/0d789c7ffe9d4498a4b15ae4d97857e969cfb16dbabb4b6f8d4a0004ec3b3c89" width="600" hegiht="" ></center>

**mixup**
mixup是Facebook人工智能研究院和MIT在“Beyond Empirical Risk Minimization”中提出的基于邻域风险最小化原则的数据增强方法，它使用线性插值得到新样本数据。令(xn,yn)是插值生成的新数据，(xi,yi)和(xj,yj)是训练集随机选取的两个数据，则数据生成方式如下：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c63c204f309b4d32a7a2c9531867f0217a21325b8eba40e1802d939e3406c1b2" width="400" hegiht="" ></center>

#### 4.2无监督的数据增强

**GAN**
GAN包含两个网络，一个是生成网络，一个是对抗网络，基本原理如下：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ae9aa492bed842b1a4aa924e232c75a04284d39025fb4738aad41be9422e15db" width="600" hegiht="" ></center>

G是一个生成图片的网络，它接收随机的噪声z，通过噪声生成图片，记做G(z)；
D是一个判别网络，判别一张图片是不是“真实的”，即是真实的图片，还是由G生成的图片。


**Autoaugmentation**
AutoAugment是Google提出的自动选择最优数据增强方案的研究，这是无监督数据增强的重要研究方向。它的基本思路是使用增强学习从数据本身寻找最佳图像变换策略，对于不同的任务学习不同的增强方法，流程如下：
(1) 准备16个常用的数据增强操作。
(2) 从16个中选择5个操作，随机产生使用该操作的概率和相应的幅度，将其称为一个sub-policy，一共产生5个sub-polices。
(3) 对训练过程中每一个batch的图片，随机采用5个sub-polices操作中的一种。
(4) 通过模型在验证集上的泛化能力来反馈，使用的优化方法是增强学习方法。
(5) 经过80~100个epoch后网络开始学习到有效的sub-policies。
(6) 之后串接这5个sub-policies，然后再进行最后的训练。

### 5.图像分类方法综述
图像分类问题，即一个输入图像，输出对该图像内容分类的描述的问题。它是计算机视觉的核心，实际应用广泛。
### 5.1 传统的图像分类方法
传统的图像分类方法流程一般为特征提取，特征编码，空间约束，图像分类。

**底层特征提取**: 通常从图像中按照固定步长、尺度提取大量局部特征描述。常用的局部特征包括SIFT(Scale-Invariant Feature Transform, 尺度不变特征转换) 、HOG(Histogram of Oriented Gradient, 方向梯度直方图) 、LBP(Local Bianray Pattern, 局部二值模式)等，一般也采用多种特征描述，防止丢失过多的有用信息。

**特征编码**: 底层特征中包含了大量冗余与噪声，为了提高特征表达的鲁棒性，需要使用一种特征变换算法对底层特征进行编码，称作特征编码。常用的特征编码方法包括向量量化编码、稀疏编码、局部线性约束编码、Fisher向量编码等。

**空间特征约束**: 特征编码之后一般会经过空间特征约束，也称作特征汇聚。特征汇聚是指在一个空间范围内，对每一维特征取最大值或者平均值，可以获得一定特征不变形的特征表达。金字塔特征匹配是一种常用的特征汇聚方法，这种方法提出将图像均匀分块，在分块内做特征汇聚。

**通过分类器分类**: 经过前面步骤之后一张图像可以用一个固定维度的向量进行描述，接下来就是经过分类器对图像进行分类。通常使用的分类器包括SVM(Support Vector Machine, 支持向量机)、随机森林等。而使用核方法的SVM是最为广泛的分类器，在传统图像分类任务上性能很好。

### 5.2深度学习图像分类方法

Alex Krizhevsky在2012年ILSVRC提出的CNN模型取得了历史性的突破，效果大幅度超越传统方法，获得了ILSVRC2012冠军，该模型被称作AlexNet。这也是首次将深度学习用于大规模图像分类中。从AlexNet之后，涌现了一系列CNN模型，不断地在ImageNet上刷新成绩，人们也在目光放在深度学习的方法上。

**ImageNet与AlexNet **

在本世纪的早期，虽然神经网络开始有复苏的迹象，但是受限于数据集的规模和硬件的发展，神经网络的训练和优化仍然是非常困难的。MNIST和CIFAR数据集都只有60000张图，这对于10分类这样的简单的任务来说，或许足够，但是如果想在工业界落地更加复杂的图像分类任务，仍然是远远不够的。

后来在李飞飞等人数年时间的整理下，2009年，ImageNet数据集发布了，并且从2010年开始每年举办一次ImageNet大规模视觉识别挑战赛，即ILSVRC。ImageNet数据集总共有1400多万幅图片，涵盖2万多个类别，在论文方法的比较中常用的是1000类的基准。

在ImageNet发布的早年里，仍然是以SVM和Boost为代表的分类方法占据优势，直到2012年AlexNet的出现。

AlexNet是第一个真正意义上的深度网络，与LeNet5的5层相比，它的层数增加了3层，网络的参数量也大大增加，输入也从28变成了224。

AlexNet有以下的特点：
* 网络比LeNet5更深，包括5个卷积层和3个全连接层。
* 使用Relu激活函数，收敛很快，解决了Sigmoid在网络较深时出现的梯度弥散问题。
* 加入了Dropout层，防止过拟合。
* 使用了LRN归一化层，对局部神经元的活动创建竞争机制，抑制反馈较小的神经元放大反应大的神经元，增强了模型的泛化能力。
* 使用裁剪翻转等操作做数据增强，增强了模型的泛化能力。预测时使用提取图片四个角加中间五个位置并进行左右翻转一共十幅图片的方法求取平均值，这也是后面刷比赛的基本使用技巧。
* 分块训练，当年的GPU计算能力没有现在强大，AlexNet创新地将图像分为上下两块分别训练，然后在全连接层合并在一起。
* 总体的数据参数大概为240M，远大于LeNet5。

**CNN**
　传统CNN包含卷积层、全连接层等组件，并采用softmax多类别分类器和多类交叉熵损失函数。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/534b7091371a41c9b38091323dd1f7b7fb48817172f84f279558e93cfd64f3c5" width="600" hegiht="" ></center>

*  卷积层(convolution layer): 执行卷积操作提取底层到高层的特征，发掘出图片局部关联性质和空间不变性质。
*  池化层(pooling layer): 执行降采样操作。通过取卷积输出特征图中局部区块的最大值(max-pooling)或者均值(avg-pooling)。降采样也是图像处理中常见的一种操作，可以过滤掉一些不重要的高频信息。 
*  全连接层(fully-connected layer，或者fc layer): 输入层到隐藏层的神经元是全部连接的。
*  非线性变化: 卷积层、全连接层后面一般都会接非线性变化函数，例如Sigmoid、Tanh、ReLu等来增强网络的表达能力，在CNN里最常使用的为ReLu激活函数。
*  Dropout: 在模型训练阶段随机让一些隐层节点权重不工作，提高网络的泛化能力，一定程度上防止过拟合。

**VGG**

牛津大学VGG(Visual Geometry Group)组在2014年ILSVRC提出的模型被称作VGG模型。该模型相比以往模型进一步加宽和加深了网络结构，它的核心是五组卷积操作，每两组之间做Max-Pooling空间降维。同一组内采用多次连续的3X3卷积，卷积核的数目由较浅组的64增多到最深组的512，同一组内的卷积核数目是一样的。卷积之后接两层全连接层，之后是分类层。

由于每组内卷积层的不同，有11、13、16、19层这几种模型，下图展示一个16层的网络结构。VGG模型结构相对简洁，提出之后也有很多文章基于此模型进行研究，如在ImageNet上首次公开超过人眼识别的模型就是借鉴VGG模型的结构。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/dfef76dea25b4413a5d6dc4cc29c3ff2c40c42a68dfc44778ce6773c71ee16b9" width="600" hegiht="" ></center>

**GoogLeNet**

GoogLeNet 在2014年ILSVRC的获得了冠军，在介绍该模型之前我们先来了解NIN(Network in Network)模型和Inception模块，因为GoogLeNet模型由多组Inception模块组成，模型设计借鉴了NIN的一些思想。

NIN模型主要有两个特点：

* 引入了多层感知卷积网络(Multi-Layer Perceptron Convolution, MLPconv)代替一层线性卷积网络。MLPconv是一个微小的多层卷积网络，即在线性卷积后面增加若干层1x1的卷积，这样可以提取出高度非线性特征。
* 传统的CNN最后几层一般都是全连接层，参数较多。而NIN模型设计最后一层卷积层包含类别维度大小的特征图，然后采用全局均值池化(Avg-Pooling)替代全连接层，得到类别维度大小的向量，再进行分类。这种替代全连接层的方式有利于减少参数。
 
 
 Inception模块如下图所示，下图左是最简单的设计，输出是3个卷积层和一个池化层的特征拼接。这种设计的缺点是池化层不会改变特征通道数，拼接后会导致特征的通道数较大，经过几层这样的模块堆积后，通道数会越来越大，导致参数和计算量也随之增大。

为了改善这个缺点，下图右引入3个1x1卷积层进行降维，所谓的降维就是减少通道数，同时如NIN模型中提到的1x1卷积也可以修正线性特征。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/7cddc83fe4c045258e48dbf16915ee1f4881fe1c7c994faabc9ab3d5b880c0a1" width="600" hegiht="" ></center>

　　GoogLeNet由多组Inception模块堆积而成。另外，在网络最后也没有采用传统的多层全连接层，而是像NIN网络一样采用了均值池化层；但与NIN不同的是，GoogLeNet在池化层后加了一个全连接层来映射类别数。

　　除了这两个特点之外，由于网络中间层特征也很有判别性，GoogLeNet在中间层添加了两个辅助分类器，在后向传播中增强梯度并且增强正则化，而整个网络的损失函数是这个三个分类器的损失加权求和。

　　GoogLeNet整体网络结构总共22层：开始由3层普通的卷积组成；接下来由三组子网络组成，第一组子网络包含2个Inception模块，第二组包含5个Inception模块，第三组包含2个Inception模块；然后接均值池化层、全连接层。
  <center><img src="https://ai-studio-static-online.cdn.bcebos.com/42b32b9cacb449c393044d72da1887dd1ffff5f70aa64925a98b6c07ee59cd8f" width="950" hegiht="" ></center>
  
**ResNet**

ResNet(Residual Network) 是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。针对随着网络训练加深导致准确度下降的问题，ResNet提出了残差学习方法来减轻训练深层网络的困难。

在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含两条路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。

残差模块如下图所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。

  <center><img src="https://ai-studio-static-online.cdn.bcebos.com/0546c6d9607c462f8867922caa6cb403afb27da235e042c5b017fa5ba6f26acc" width="400" hegiht="" ></center>

下图展示了50、101、152层网络连接示意图，使用的是瓶颈模块。
  <center><img src="https://ai-studio-static-online.cdn.bcebos.com/661860b3b9d74eaaa9986c9eec6eeb68c4751d510ee6492dbf9084a1c3e54451" width="800" hegiht="" ></center>

  
  
  





