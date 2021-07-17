# 可形变卷积（Deformable Convolution）V1


## 提出背景和基础概念——deformable

卷积神经网络在视觉识别任务中取得了显著的成功，其对几何变换建模的能力主要来自广泛的数据扩充、巨大的模型容量和一些简单的手工制作的模块。

但是，CNNs天生局限于模拟大型的未知转换。**局限性源于CNN模块的固定几何结构**:卷积单元在固定位置对输入特征图进行采样；池化层以固定比率降低空间分辨率；RoI(感兴趣区域)汇集层将RoI分成固定的空间箱等。

缺乏处理几何变换的内部机制，常常会导致问题。

**可形变卷积加强CNN对形变的建模能力**，它通过学习一个额外的偏移（offset），使卷积核对输入feature map的采样的产生偏移，集中于感兴趣的目标区域，使采样网格能够自由变形。如下图所示。通过额外的卷积层，从前面的特征映射中学习偏移。因此，变形以局部、密集和自适应的方式取决于输入特征。

![](https://ai-studio-static-online.cdn.bcebos.com/48679208bbc44d58b33f2c6b008bb682d8ff1157b97e400e9b52bf9996c916a5)

3×3标准和可变形卷积中采样位置的图示。(a)常规采样网格(绿点)的标准卷积。(b)在可变形卷积中具有增大偏移(浅蓝色箭头)的变形采样位置(深蓝色点)。(c)(d)是(b)的特例，表明可变形卷积概括了比例、(各向异性)纵横比和旋转的各种变换。**箭头表示卷积核权重的偏移。图c为可变形卷积学到了平移尺度形变，图d为旋转形变。**

基于同样的思想，可形变池化也被提出。可形变池化和可形变卷积这两个轻量级模块，为偏移学习增加了少量的参数和计算。它们可以很容易地在深层中枢神经系统中取代普通的同类，并且可以很容易地通过标准反向传播进行端到端的训练。由此产生的神经网络被称为可变形卷积网络。


## 原理——Deformable Convolution
可变形卷积很好理解，但如何实现呢？实现方面需要关注两个限制：

1、如何将它变成单独的一个层，而不影响别的层；

2、在前向传播实现可变性卷积中，如何能有效地进行反向传播。

这两个问题的答案分别是：

1、在实际操作时，并不是真正地把卷积核进行扩展，而是**对卷积前图片的像素重新整合**，

变相地实现卷积核的扩张；

2、在图片像素整合时，需要对像素进行偏移操作，偏移量的生成会产生浮点数类型，而偏移量又必须转换为整形，直接对偏移量取整的话无法进行反向传播，这时采用**双线性差值**的方式来得到对应的像素。

用数学语言阐述如下：
在二维卷积中，输入特征图x，输出特征图y，w作为采样权值，R定义了网格空间中感受野的大小和扩张，将之和具体的某一个坐标点对应得到该坐标点的临近坐标群，从而可以帮助定义卷积等操作：

如：![](https://ai-studio-static-online.cdn.bcebos.com/f90468e7cf074187af0e8656f0219c9a17e753ecc9334289890ccee7b185dd81)

标准卷积：

对于输出特征地图y上的每个位置P0，![](https://ai-studio-static-online.cdn.bcebos.com/53710d6768b44154b214451555c6c95a1cd01798f28448aa8441c8017e40d7fd)

其中pn枚举R中的位置，即对应权值w和对应x积和赋给y（p0）


可形变卷积：

和标准卷积相比，此处提出偏移量：![](https://ai-studio-static-online.cdn.bcebos.com/6f8dcf14b036424ca57e26cfe71f57cc1b8d823d0dd44e25a6974287e0c5e17d)

则卷积公式：![](https://ai-studio-static-online.cdn.bcebos.com/22fa13650e1740a89e30f42e1eeb3dd7932bf64e9a974fb1b0ecac0ea6acfa5e)

偏移量Δpn通常是小数，然而在图像空间中，只有网格点上有像素值，对于一个非网格点位置求像素值，则需要使用双线性插值（原理可见 https://blog.csdn.net/qq_38701868/article/details/103511663）

即：![](https://ai-studio-static-online.cdn.bcebos.com/4ab15f3e0764470d90dadca3981e98f86baa440b1e704cc6b6e94040327e7f76)

p = p0+pn+ ∆pn，q枚举特征图x中的所有整数空间位置，G(.,.)为双线性插值核。该公式计算速度很快，因为G(q，p)只对少数几个qs是非零的。

如下所示，3*3可变形卷积
![](https://ai-studio-static-online.cdn.bcebos.com/be0beb4edef94b319cbb0599fa8357e54ed2e46d2c9448aa872f7588cc2b1926)

偏移是通过在同一输入特征图上应用卷积层获得的。卷积核具有与当前卷积层相同的空间分辨率和膨胀。输出偏移场具有与输入要素图相同的空间分辨率。信道维度2N对应于N个2D偏移。在训练期间，同时学习用于生成输出特征和偏移的卷积核。为了学习偏移，梯度通过以上等式反向传播。


##  算法流程

可变性卷积的流程为：

1、原始图片batch（大小为b*h*w*c），记为U，经过一个普通卷积，卷积填充为same，即输出输入大小不变，对应的输出结果为（b*h*w*2c)，记为V，输出的结果是指原图片batch中每个像素的偏移量（x偏移与y偏移，因此为2c）。

2、将U中图片的像素索引值与V相加，得到偏移后的position（即在原始图片U中的坐标值），需要将position值限定为图片大小以内。

position的大小为（b*h*w*2c)，但position只是一个坐标值，而且还是float类型的，我们需要这些float类型的坐标值获取像素。

3、例，取一个坐标值（a,b)，将其转换为四个整数，floor(a), ceil(a), floor(b), ceil(b)，将这四个整数进行整合，

得到四对坐标（floor(a),floor(b)),  ((floor(a),ceil(b)),  ((ceil(a),floor(b)),  ((ceil(a),ceil(b))。这四对坐标每个坐标都对应U中的一个像素值，而我们需要得到(a,b)的像素值，这里采用双线性差值的方式计算

（一方面得到的像素准确，另一方面可以进行反向传播）。

4、在得到position的所有像素后，即得到了一个新图片M，将这个新图片M作为输入数据输入到别的层中，如普通卷积。

## 作用 及应用场景
可变形卷积单元具有诸多良好的性质。它不需要任何额外的监督信号，可以直接通过目标任务学习得到。它可以方便地取代任何已有视觉识别任务的卷积神经网络中的若干个标准卷积单元，并通过标准的反向传播进行端到端的训练。是对于传统卷积网络简明而又意义深远的结构革新，具有重要的学术和实践意义。它适用于所有待识别目标具有一定几何形变的任务（几乎所有重要的视觉识别任务都有此特点，人脸、行人、车辆、文字、动物等），可以直接由已有网络结构扩充而来，无需重新预训练。它仅增加了很少的模型复杂度和计算量，且显著提高了识别精度。例如，在用于自动驾驶的图像语义分割数据集（CityScapes）上，可变形卷积神经网络将准确率由70%提高到了75%。

参考：
【1】https://blog.csdn.net/mykeylock/article/details/77746499

【2】Deformable Convolutional Networks

【3】https://blog.csdn.net/yeler082/article/details/78370795

运行实例：


```python
import paddle
input = paddle.rand((8, 1, 28, 28))
kh, kw = 3, 3
# offset shape should be [bs, 2 * kh * kw, out_h, out_w]
# mask shape should be [bs, hw * hw, out_h, out_w]
# In this case, for an input of 28, stride of 1
# and kernel size of 3, without padding, the output size is 26
offset = paddle.rand((8, 2 * kh * kw, 26, 26))
deform_conv = paddle.vision.ops.DeformConv2D(
    in_channels=1,
    out_channels=16,
    kernel_size=[kh, kw])
out = deform_conv(input, offset)
print(out.shape)
# returns
#[8, 16, 26, 26]
```

    [8, 16, 26, 26]


# 形变卷积（Deformable Convolution）V2

V2说V1存在的问题是在RoI外部的这种几何变化适应性表现得不好，导致特征会受到无关的图像内容影响（this support may nevertheless extend well beyond the region of interest，causing features to be influenced by irrelevant image content）。

作为一个新版本的可变形卷积网络，称为可变形卷积网络v2 (DCNv2)，具有增强的学习可变形卷积的建模能力。建模能力的提高有两种互补的形式。首先是网络中**可变形卷积层的扩展使用。** 为更多卷积层配备偏移学习能力，使DCNv2能够在更宽的特征级别范围内控制采样。第二种是**可变形卷积模块中的调制机制**，其中每个样本不仅经历学习偏移，而且还被学习特征幅度调制。因此，网络模块能够改变其样本的空间分布和相对影响。

**DCN v1中引入的offset是要寻找有效信息的区域位置，DCN v2中引入modulation是要给找到的这个位置赋予权重，这两方面保证了有效信息的准确提取**

## 概念

有效感受野（Effective receptive fields）：网络中每个节点都会计算feature map的一个像素点，而这个点就有它自己的感受野，但是不是感受野中的所有像素对这个点的响应的贡献都是相同的，大小与卷积核权重有关，因此文中用有效感受野来表示这种贡献的差异。

有效采样/bin位置（Effective sampling/bin locations）：对于卷积核的采样点和RoIpooling的bin的位置进行有助于理解DCN，有效位置在反应采样点位置的基础上还反应了每个位置的贡献。

错误边界显著性区域（Error-bounded saliency regions）：最近关于图像显著性的研究表明，对于网络的每个节点的响应，不是图像上所有的区域对其都有影响，去掉一些不重要的区域，节点的响应可以保持不变。根据这一性质，文章将每个节点的support region限制到了最小的可以和整幅图产生相同的响应的区域，并称之为错误边界显著性区域。

##  原理
给定K个采样位置的卷积核，让wk和PK分别表示第K个位置的权重和预先指定的偏移量。例如，K = 9，Pk∈{(1，1)，(1，0)，……，(1，1)}定义了膨胀为1的3 × 3卷积核。让x(p)和y(p)分别表示来自输入特征图x和输出特征图sy的位置p处的特征。然后，调制的可变形卷积可以表示为：

![](https://ai-studio-static-online.cdn.bcebos.com/eb59590ef4494dc0827066f4c44f57f45a872c0f9af2450890427587bc91f7dd)

其中 Δpk 和 Δmk 分别为第k个位置的可学习偏移和调制标量。在deformable conv v1中 Δmk 为1。可见v1其实可以看做是v2的特例



## 作用及应用场景
![](https://ai-studio-static-online.cdn.bcebos.com/0cbaf618049941c99015676fa9cd3f5792ddb536bab747f99dd387924f384ccc)

1、基于常规卷积层的深度网络对于形变目标有一定的学习能力，比如(a)中的最后一行，基本上都能覆盖对应的目标区域或者非目标区域，这主要归功于深度网络的拟合能力，这种拟合能力有点强行拟合的意思，所以才有DCN这种设计。

2、DCNv1对于形变目标的学习能力要比常规卷积强，能够获取更多有效的信息。比如(b)中的最后一行，当输出点位置在目标上时（前2张图），影响区域相比常规卷积而言更大。

3、DCNv2对于形变目标的学习能力比DCNv1更强，不仅能获取更多有效的信息，而且获取的信息更加准确，比如©中的最后一行，目标区域更加准确。因此简单来讲，DCNv1在有效信息获取方面的recall要高于常规卷积，而DCNv2不仅有较高的recall，而且有较高的precision，从而实现信息的精确提取。


参考
【1】：https://blog.csdn.net/qq_37014750/article/details/84659473

【2】：Deformable ConvNets v2: More Deformable, Better Results

【3】：https://blog.csdn.net/u014380165/article/details/88072737

应用实例：


```python
#deformable conv v2:

import paddle
input = paddle.rand((8, 1, 28, 28))
kh, kw = 3, 3
# offset shape should be [bs, 2 * kh * kw, out_h, out_w]
# mask shape should be [bs, hw * hw, out_h, out_w]
# In this case, for an input of 28, stride of 1
# and kernel size of 3, without padding, the output size is 26
offset = paddle.rand((8, 2 * kh * kw, 26, 26))
mask = paddle.rand((8, kh * kw, 26, 26))
deform_conv = paddle.vision.ops.DeformConv2D(
    in_channels=1,
    out_channels=16,
    kernel_size=[kh, kw])
out = deform_conv(input, offset, mask)
print(out.shape)
# returns
#[8, 16, 26, 26]
```

    [8, 16, 26, 26]

