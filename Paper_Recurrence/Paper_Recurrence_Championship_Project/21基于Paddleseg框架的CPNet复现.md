#  一、论文简介

> 论文原文：[2004.01547.pdf ](https://arxiv.org/pdf/2004.01547.pdf)



### 简介

这篇文章主要讨论了上下文先验在场景分割的应用，发表于CVPR2020。

- **Motivation**

由于卷积层结构的限制，FCN提供的上下文信息不足，有待改进。因此，近年来各种方法被提出来用于探索上下文依赖性，以获得更准确的分割结果。目前为止，上下文信息的聚合主要有两种方法:

**基于金字塔的方法：**
有几种方法如PSPNet采用基于金字塔的模块或全局池化来有规律的聚合区域或全局上下文信息。然而，它们捕获了同类的上下文关系，却忽略了不同类别的上下文。如图1(b)示。当场景中有混淆的类别时，这些方法可能导致不太可靠的上下文。

**基于注意力的方法：**
最近基于注意力的方法，如通道注意力、空间注意力，有选择地聚合不同类别之间的上下文信息。然而，由于缺乏明确的区分，注意力机制的关系描述不够清晰。因此，它可能会选择不需要的上下文依赖关系，如图1(e)所示。

![图1：语义分割很难分割的情形举例 。在第一排，红色方框中沙子的中心部分被错误的归类为海，因为阴影部分与海有着相似的外观。使用基于金字塔的方法，对混乱的空间信息进行聚合，可能会导致(b)中所显示的不良预测。在第二行中，绿色方框中的桌子与床的底部外观相似。基于注意力的方法在没有先验知识的情况下无法有效区分混乱的空间信息，导致(e)所示的预测准确率较低。CPNet是作者提出的方法。](https://ss.im5i.com/2021/10/10/lMFGG.jpg)

这两种方法聚合上下文信息，没有明显的区别类内和类间上下文 ，导致了不同上下文关系的混合。

为了去解决这些问题，在这篇文章中，先提出一个**Context Prior**去对相同类(**intra-context**)的像素与不同类(**inter-context**)的像素之间的关系进行建模。基于**Context Prior**，提出了**Context Prior Network**，它包含了受**Affinity Loss**监督的**Context Prior Layer**。

### Affinity Loss

为了去明确规范这个网络去学习类别之间的关系，我们引入了Affinity Loss。对于图像上的每一个像素，**Affinity Loss将会迫使网络去考虑在同一类别的像素和不同类别的像素**。

![](https://ss.im5i.com/2021/10/10/l7O24.jpg)

只要给定输入的**Groundtruth**，我们就可以知道每一个像素的`Context Prior`（哪些属于同一类，哪些属于不同类），可以学习`Context Prior`去引导整个网络。首先根据**ground truth**建立**Ideal Affinity Map**作为监督：

给定输入图像$I$和其对应的groundtruth$L$，将 $I$ 送入网络中可以得到一个大小为$H\times W$的特征图$X$。我们先将$L$下采样到与$X$大小相同，得到一个更小的groundtruth $\tilde{L}$。我们使用了 $one-hot$ 编码去对$\tilde{L}$中的类别标签进行编码，得到一个大小为 $H\times W\times C$ 的矩阵 $\hat{L}$ ， $C$为类别数。紧接着，将 $L$ Reshape成 $N\times C$ 大小， $N=H\times W$。最后，我们进行一个矩阵乘法：$A = L\times L^{T}$，这个A就是我们期望的$Ideal \ Affinity \ Map$，大小为$N\times N$。 我们使用$Ideal\ Affinity\ Map$去监督$Context\ Prior\ Map$的生成。

对于$Context\ Prior\ Map$中的每一个像素，其实是一个二分类问题。通常解决二分类问题用到的是二元交叉熵损失：
$$
\mathcal{L}\_{u}=-\frac{1}{N^{2}}(a_{n}\log{p_{n}}+(1-a_{n})\log{(1-p_{n})}).
$$

其中$\{p_{n}\in P,n\in [1,,N^{2}]\}$，$P$为预测得到的$Prior \ map$，$\{a_{n}\in A,n\in [1,,N^{2}]\}$，$A$为期望得到的$Ideal\ Affinity\ Map$。

> ❝ However, such a unary loss only considers the isolated pixel in the prior map ignoring the semantic correlation with other pixels.

但是它只考虑了在$Prior \ map$中的单个像素，忽略了与其他像素的语义联系。所以需要另外一部分损失。我们可以在$A$中看到每一行其实都对应着feature map X中的每个像素，可以它们表示为同类像素或者不同类像素，这个关系是有助于推理语义相关性和场景结构的。因此，我们考虑把同类像素和不同类像素作为两个部分来对关系分别编码：

$$\begin{gathered}
\mathcal{T}^{p}_{j}=\log\frac{\sum_{i=1}^{N}a_{ij}p_{ij}}{\sum_{i=1}^{N}p_{ij}},\\
\mathcal{T}^{r}_{j}=\log\frac{\sum_{i=1}^{N}a_{ij}p_{ij}}{\sum_{i=1}^{N}a_{ij}},\\
\mathcal{T}^{s}_{j}=\log\frac{\sum_{i=1}^{N}(1-a_{ij})(1-p_{ij})}{\sum_{i=1}^{N}(1-a_{ij})},\\
\mathcal{L}_{g}=-\frac{1}{N}\sum_{j=1}^{N}(\mathcal{T}^{p}_{j}+\mathcal{T}^{r}_{j}+\mathcal{T}^{s}_{j})
\end{gathered}$$

其中，$\mathcal{T}_{j}^{p}$、$\mathcal{T}_{j}^{r}$、$\mathcal{T}_{j}^{s}$分别表示$P$中第$j$行的同类准确率(precision)，同类召回率(recall)，特异度(specificity)。

最后，完整的Affinity Loss定义如下：
$$
\mathcal{L}\_{p}=\lambda_{u}\mathcal{L}_{u}+\lambda_{g}\mathcal{L}_{g}.
$$
其中，$\lambda_{u}$和$\lambda_{g}$用于平衡一元损失(unary loss)和全局损失(global loss)，根据经验，通常都是直接设为1。

### 网络结构(Context Prior Layer)

![](https://ss.im5i.com/2021/10/10/l7pkR.jpg)

作者设计上下文先验层（**Context Prior Layer**），它可以在任何backbone的最后插入以进行语义分割的任务。**Context Prior Layer**使用backbone如resnet50输出的特征图（图中最左边的黄色立体）作为输入特征$X$，形状为$H\times W\times C_{0}$。经过聚合模块 `Aggregation`聚合局部上下文信息之后，得到$\tilde{X}$，送入一个$1\times1$卷积层+BN+sigmoid层进行处理，形状变为$H\times  W\times N(N=H\times W)$，最后进行reshape得到了`Context Prior Map`。`Context Prior Map`由`GT`产生的`Ideal Affinity Map`进行监督，学习得到了类内的上下文信息。将他与经过聚合后的特征图相乘后得到富含类内上下文信息的特征图。与此同时，用1-P得到可以得到类间上下文，同样的操作可以得到富含类间上下文信息的特征图。将原图和两种上下文Concat去输出最终的预测$F=Concat(X,Y,Y^{-})$

### Aggregation Module

**Context Prior Map**需要一些局部空间信息去推理语义相关性，一般来说要聚合更多的空间信息就要使用更大的卷积核，但是计算代价是很大的。因此，作者用全分离卷积(**fully separable convolution**)设计了一个有效的**Aggregation Module**去聚合空间信息。

![](https://ss.im5i.com/2021/10/10/l7Mld.jpg)

我们将一个普通卷积分解成两个在空间上不对称的卷积，比如一个$k\times k$的卷积，我们这样来替代：

先使用一个$k\times 1$卷积，接着再使用一个$1\times k$卷积，我们称为空间可分离卷积($spatial\ separable\ convolution$)，并且每个空间卷积都采用深度卷积($depth-wise convolution$)，这导致计算量大大减少。由于考虑了空间和深度两个维度，作者把这种方式称为$Fully\ Separable\ Convolution$。

### Network Architecure

Context Prior Network(CPNet)是一个全卷积网络，由Backbone和一个Context Prior Layer组成。Backbone是一个用了空洞卷积(dilation strategy)的现成网络。同时，在backbone网络的阶段4(stage 4)还使用了辅助损失(auxiliary loss)，也是一个交叉熵损失。最终的损失函数为：
$$
\mathcal{L}=\lambda_{s}\mathcal{L}_{s}+\lambda_{a}\mathcal{L}_{a}+\lambda_{p}\mathcal{L}_{p}
$$
其中$\mathcal{L}_{s}$, $\mathcal{L}_{a}$，$\mathcal{L}_{p}$ 分别代表主要分割损失(main segmentation loss)，辅助损失(auxiliary loss)以及Affinity Loss，$λs$, $λa$, $λp$ 是用来平衡loss的权重。经验性地设置为：$λs=1, λa=1, λp=0.4$。

### 实验结果

在ADE20K上测试

![](https://ss.im5i.com/2021/10/10/lMhA5.jpg)

在Cityscape上测试

![](https://ss.im5i.com/2021/10/10/lMQ5z.jpg)

# 二、数据集介绍

使用的数据集为：[ADE20K官网](https://groups.csail.mit.edu/vision/datasets/ADE20K/)

ADE20k 拥有超过 25,000 张图像（20ktrain，2k val，3ktest），这些图像用开放字典标签集密集注释。对于 2017 Places Challenge 2，选择了覆盖 89％所有像素的 100 个 thing 和 50 个 stuff 类别。在本研究中作者使用后者这个封闭的词汇表。

- Training set：20.210 images
- Validation set：2.000 images

![](https://ss.im5i.com/2021/10/10/lMCq6.jpg)

> 注：paddleseg 提供 ADE20K 在线下载以及解压，详情请看[paddleseg 开发文档](https://paddleseg.readthedocs.io/zh_CN/release-2.1/apis/transforms/datasets.html#ADE20K)

# 项目介绍

本项目基于PaddlePaddle框架复现CPNet，项目依赖于paddleseg工具，因此可以使用paddleseg中提供的训练和评估API进行训练与评估。

## 一、复现结果

| Model                 | mIOU  |
| --------------------- | ----- |
| CPNet50(原论文 mmseg) | 44.46 |
| CPNet50(PaddelSeg)        | 45.37 |

## 二、预训练模型

在`work/openContext/output/best_model`为训练最好结果
| 文件名                 | 说明  |
| --------------------- | ----- |
|  model.pdparams       | 模型文件 |

## 三、文件结构

```
├── models/ # 模型定义
├── loss/ # 自定义损失函数
├── tool/ # 工具
├── train.py # 训练
├── eval.py # 评估
```

# 开始实践

## 一、下载安装PaddleSeg依赖


```python
!pip install paddleseg --user
```

## 二、使用预训练模型验证


```python
%cd /home/aistudio/work/openContext/
!python eva.py
```

## 三、训练


```python
%cd /home/aistudio/work/openContext/
!python train.py
```

## 代码说明

该论文主要是损失函数的构建

```
class AffinityLoss(nn.Layer):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(AffinityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        # 设置类别数，若使用Cityscape训练需要修改为19！！！
        self.num_classes = 150

    # One_hot编码得到ideal_Affinity_map
    def _construct_ideal_affinity_matrix(self, label, label_size):
        label = paddle.unsqueeze(label, axis=1)
        # 下采样到相同大小
        scaled_labels = F.interpolate(
            label, size=label_size, mode="nearest")

        scaled_labels = scaled_labels.squeeze_().astype('int64')
        scaled_labels[scaled_labels==255]=self.num_classes
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)

        one_hot_labels = one_hot_labels.reshape((one_hot_labels.shape[0],
                                                 -1, self.num_classes + 1))

        ideal_affinity_matrix = paddle.bmm(one_hot_labels,
                                           one_hot_labels.transpose((0, 2, 1)))
        return ideal_affinity_matrix

    def forward(self, cls_score, label):
    	# 60为context prior map大小
        ideal_affinity_matrix = self._construct_ideal_affinity_matrix(label, [60, 60])

        unary_term = F.binary_cross_entropy(cls_score, ideal_affinity_matrix)

        diagonal_matrix = (1 - paddle.eye(ideal_affinity_matrix.shape[1]))
        vtarget = paddle.multiply(diagonal_matrix, ideal_affinity_matrix)

        recall_part = paddle.sum(cls_score * vtarget.squeeze(), axis=2)
        denominator = paddle.sum(ideal_affinity_matrix, axis=2)
        # 很重要，去除负数部分，因做分母
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)

        recall_part = paddle.divide(recall_part, denominator)
        recall_label = paddle.ones_like(recall_part)
        recall_loss = F.binary_cross_entropy(recall_part, recall_label)


        spec_part = paddle.sum(paddle.multiply(1 - cls_score,1 - ideal_affinity_matrix), axis=2)

        denominator = paddle.sum(1 - ideal_affinity_matrix, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        spec_part = paddle.divide(spec_part, denominator)
        spec_label = paddle.ones_like(spec_part)
        spec_loss = F.binary_cross_entropy(spec_part, spec_label)


        precision_part = paddle.sum(paddle.multiply(cls_score, ideal_affinity_matrix), axis=2)
        denominator = paddle.sum(cls_score, axis=2)
        denominator = paddle.where(denominator <= 0, paddle.ones_like(denominator), denominator)
        precision_part = paddle.divide(precision_part, denominator)
        precision_label = paddle.ones_like(precision_part)
        precision_loss = F.binary_cross_entropy(precision_part, precision_label)


        global_term = (recall_loss + spec_loss + precision_loss) / 60
        loss_cls = unary_term + global_term
        return loss_cls

```

模型部分


```python
class CPNet(nn.Layer):
    def __init__(self, prior_channels, proir_size, am_kernel_size, pretrained=None, groups=1, ):
        super().__init__()

        # context proir map通道数
        self.in_channels = 2048
        self.channels = 256
        self.backbone = ResNet_vd(50, pretrained=pretrained)

        self.prior_channels = prior_channels
        self.prior_size = paddle.to_tensor([proir_size, proir_size])
        self.aggregation = AggregationModule(self.in_channels, prior_channels,
                                             am_kernel_size)
        self.prior_conv = nn.Sequential(
            nn.Conv2D(
                self.prior_channels,
                paddle.prod(self.prior_size),
                1,
                padding=0,
                groups=groups), nn.BatchNorm2D(paddle.prod(self.prior_size)))
        self.intra_conv = ConvBNReLU(
            self.prior_channels, self.prior_channels, 1, padding=0, stride=1)
        self.inter_conv = ConvBNReLU(
            self.prior_channels,
            self.prior_channels,
            1,
            padding=0,
            stride=1,
        )

        self.bottleneck = ConvBNReLU(
            self.in_channels + self.prior_channels * 2,
            self.channels,
            3,
            padding=1,
        )
        self.aux = AuxLayer(
            in_channels=self.in_channels,
            out_channels=150, inter_channels=self.channels)

    def forward(self, inputs):
        # inputs B H w C_0
        H = inputs.shape[2]
        W = inputs.shape[3]
        # 验证时图大小不一致，进行放缩
        if H != 480 or W != 480:
            inputs = F.interpolate(inputs, (480, 480),
                                   mode='bilinear',
                                   align_corners=True)
        conv1, conv2, conv3, conv4 = self.backbone(inputs)
        batch_size, channels, height, width = conv4.shape
        assert self.prior_size[0] == height and self.prior_size[1] == width

        value = self.aggregation(conv4)

        context_prior_map = self.prior_conv(value)

        context_prior_map = context_prior_map.reshape(
            (batch_size, paddle.prod(self.prior_size), -1))

        context_prior_map = paddle.transpose(context_prior_map, (0, 2, 1))
        context_prior_map = F.sigmoid(context_prior_map)
        inter_context_prior_map = 1 - context_prior_map


        value = value.reshape((batch_size, self.prior_channels, -1))

        value = paddle.transpose(value, (0, 2, 1))


        intra_context = paddle.bmm(context_prior_map, value)

        intra_context = intra_context / paddle.prod(self.prior_size)

        intra_context = intra_context.transpose((0, 2, 1))
        intra_context = intra_context.reshape((batch_size, self.prior_channels,
                                               self.prior_size[0],
                                               self.prior_size[1]))
        intra_context = self.intra_conv(intra_context)

        inter_context = paddle.bmm(inter_context_prior_map, value)
        inter_context = inter_context / paddle.prod(self.prior_size)
        inter_context = inter_context.transpose((0, 2, 1))
        inter_context = inter_context.reshape((batch_size, self.prior_channels,
                                               self.prior_size[0],
                                               self.prior_size[1]))
        inter_context = self.inter_conv(inter_context)

        cp_outs = paddle.concat([conv4, intra_context, inter_context], axis=1)
        output = self.bottleneck(cp_outs)
        output = F.interpolate(output, (H, W),
                               mode='bilinear',
                               align_corners=True)
        # 辅助Loss头
        aux = self.aux(conv4)
        aux = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=True)
        return output, aux, context_prior_map
```



请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
