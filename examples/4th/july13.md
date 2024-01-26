##1、目标检测任务解决正负样本不均衡的手段有哪些？
正样本是指图片中感兴趣的目标区域；负样本指目标区域之外的背景区域。在训练时，正样本又可分为易分正样本(容易正确分类的正样本)和难分正样本(容易被错分为负样本的正样本)；同理负样本也可分为易分负样本和难分负样本。
易分样本对模型来说是一个简单样本，模型很难从这个样本中得到更多的信息，单个易分样本的损失函数较小，损失函数对输入的梯度的幅值也相对较小(后续会有解释)。难分样本对模型来说是一个困难的样本，它产生的梯度信息则会更丰富，它更能指导模型优化的方向。
样本不均衡问题：指在训练的时候各个类别的样本数量不均衡。以基于深度学习的目标检测为例，样本类别不均衡主要体现在两方面：正负样本不均衡和难易样本不均衡。一般在目标检测任务框架中，保持正负样本的比例为1:3（经验值）。
**OHEM（online hard example mining，发表于CVPR 2016）**
OHEM算法，主要是针对训练过程中的困难样本自动选择，其核心思想是根据输入样本的损失进行筛选，筛选出困难样本（即对分类和检测影响较大的样本），然后将筛选得到的这些样本应用在随机梯度下降中训练。传统的Fast RCNN系列算法在正负样本选择的时候采用当前RoI与真实物体的IoU阈值比较的方法，这样容易忽略一些较为重要的难负样本，并且固定了正、负样本的比例与最大数量，显然不是最优的选择。以此为出发点，OHEM将交替训练与SGD优化方法进行了结合，在每张图片的RoI中选择了较难的样本，实现了在线的难样本挖掘。
![Alt text](./1.1.png)
OHEM实现在线难样本挖掘的网络如上图所示。图中包含了两个相同的RCNN网络，上半部的a部分是只可读的网络，只进行前向运算；下半部的b网络即可读也可写，需要完成前向计算与反向传播。
当然，为了实现方便，OHEM的简单实现可以是：在原有的Fast-RCNN里的loss layer里面对所有的props计算其loss，根据loss对其进行排序，选出K个hard examples，反向传播时，只对这K个props的梯度/残差回传，而其他的props的梯度/残差设为0。
但是，由于其特殊的损失计算方式，把简单的样本都舍弃了，导致模型无法提升对于简单样本的检测精度，这也是OHEM方法的一个弊端。

**S-OHEM（Stratified Online Hard Example Mining，发表于CCCV 2017）**
在OHEM中定义的多任务损失函数(包括分类损失和定位损失)，在整个训练过程中各类损失具有相同的权重，这种方法忽略了训练过程中不同损失类型的影响，例如在训练期的后期，定位损失更为重要，因此OHEM缺乏对定位精度的足够关注。 因此S-OHEM根据loss的分布抽样训练样本。
作者采用了stratified sampling的方法，首先根据分类损失和定位损失的比率将RoIs分为四组，分组时，每个组依据的propositional formula和 the required sample size都是随着训练过程而动态变化的。在每个分组内，通过对loss的排序来选择hard eamples。之后，根据动态分布对RoIs进行二次采样，最终得到B个困难样本。
![Alt text](./1.2.png)

**GHM（gradient harmonizing mechanism，发表于AAAI 2019）**
作者指出难度不同样本的不均衡性可以在梯度模长的分布上体现出来。通过对梯度分布的研究，作者提出了一种梯度均衡策略GHM可以有效地改进单阶段检测器的性能。作者提出了梯度密度，并将梯度密度的倒数作为损失函数的权重分别引入到分类损失函数(GHM-C)和边框损失函数(GHM-R)中。
梯度模长反映了样本的分类困难程度和它对整体梯度的贡献，解释如下图：
![Alt text](./1.3.jpg)

下图是一个收敛模型的梯度模长的分布，可以看出简单样本的数量很大，使得它对梯度的整个贡献很大，另一个需要的地方是，在梯度模较大的地方仍然存在着一定数量的分布，说明模型很难正确处理这些样本，作者把这类样本归为离群样本，因为他们的梯度模与整体的梯度模的分布差异太大，并且模型很难处理，如果让模型强行去学习这些离群样本，反而会导致整体性能下降。
![Alt text](./1.4.jpg)

##2、对比YOLOv1、YOLOv2、YOLOv3、YOLOv4、YOLOv5五个模型
**YOLOv1**
核心思想：将整张图片作为网络的输入（类似于Faster-RCNN），直接在输出层对BBox的位置和类别进行回归。
将一幅图像分成SxS个网格(grid cell)，如果某个object的中心 落在这个网格中，则这个网格就负责预测这个object。
每个网络需要预测B个BBox的位置信息和confidence（置信度）信息，一个BBox对应着四个位置信息和一个confidence信息。confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息：
![Alt text](./2.1.png)
其中如果有object落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。
每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5*B+C)的一个tensor。（注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。）
损失函数：sum-squared error loss
![Alt text](./2.2.jpg)
优点：
1快速，pipline简单.
2背景误检率低。
3通用性强。YOLO对于艺术类作品中的物体检测同样适用。它对非自然图像物体的检测率远远高于DPM和RCNN系列检测方法。
缺点
1 由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。
2 虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。
3 YOLO loss函数中，大物体IOU误差和小物体IOU误差对网络训练中loss贡献值接近（虽然采用求平方根方式，但没有根本解决问题）。因此，对于小物体，小的IOU误差也会对网络优化过程造成很大的影响，从而降低了物体检测的定位准确性。

**YOLOv2**
YOLOv2相对v1版本，在继续保持处理速度的基础上，从预测更准确（Better），速度更快（Faster），识别对象更多（Stronger）这三个方面进行了改进。其中识别更多对象也就是扩展到能够检测9000种不同对象，称之为YOLO9000。
文章提出了一种新的训练方法–联合训练算法，这种算法可以把这两种的数据集混合到一起。使用一种分层的观点对物体进行分类，用巨量的分类数据集数据来扩充检测数据集，从而把两种不同的数据集混合起来。
联合训练算法的基本思路就是：同时在检测数据集和分类数据集上训练物体检测器（Object Detectors ），用检测数据集的数据学习物体的准确位置，用分类数据集的数据来增加分类的类别量、提升健壮性。
YOLO9000就是使用联合训练算法训练出来的，他拥有9000类的分类信息，这些分类信息学习自ImageNet分类数据集，而物体位置检测则学习自COCO检测数据集。
改进：
    1、Batch Normalization（批量归一化）
mAP提升2.4%。
2、High resolution classifier（高分辨率图像分类器）
mAP提升了3.7%。
3、Convolution with anchor boxes（使用先验框）
召回率大幅提升到88%，同时mAP轻微下降了0.2。


**YOLOv3**
YOLO v3的模型比之前的模型复杂了不少，可以通过改变模型结构的大小来权衡速度与精度。
简而言之，YOLOv3 的先验检测（Prior detection）系统将分类器或定位器重新用于执行检测任务。他们将模型应用于图像的多个位置和尺度。而那些评分较高的区域就可以视为检测结果。此外，相对于其它目标检测方法，我们使用了完全不同的方法。我们将一个单神经网络应用于整张图像，该网络将图像划分为不同的区域，因而预测每一块区域的边界框和概率，这些边界框会通过预测的概率加权。我们的模型相比于基于分类器的系统有一些优势。它在测试时会查看整个图像，所以它的预测利用了图像中的全局信息。与需要数千张单一目标图像的 R-CNN 不同，它通过单一网络评估进行预测。这令 YOLOv3 非常快，一般它比 R-CNN 快 1000 倍、比 Fast R-CNN 快 100 倍。

改进之处
1、多尺度预测 （引入FPN）。
2、更好的基础分类网络（darknet-53, 类似于ResNet引入残差结构）。
3、分类器不在使用Softmax，分类损失采用binary cross-entropy loss（二分类交叉损失熵）
YOLOv3不使用Softmax对每个框进行分类，主要考虑因素有两个：
1、Softmax使得每个框分配一个类别（score最大的一个），而对于Open Images这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类。
2、Softmax可被独立的多个logistic分类器替代，且准确率不会下降。

多尺度预测：
每种尺度预测3个box, anchor的设计方式仍然使用聚类,得到9个聚类中心,将其按照大小均分给3个尺度.
尺度1: 在基础网络之后添加一些卷积层再输出box信息.
尺度2: 从尺度1中的倒数第二层的卷积层上采样(x2)再与最后一个16x16大小的特征图相加,再次通过多个卷积后输出box信息.相比尺度1变大两倍.
尺度3: 与尺度2类似,使用了32x32大小的特征图. 

**YOLOv4**
YOLOv4 在COCO上，可达43.5％ AP，速度高达 65 FPS！
YOLOv4的特点是集大成者，俗称堆料。但最终达到这么高的性能，一定是不断尝试、不断堆料、不断调参的结果，给作者点赞。下面看看堆了哪些料：
Weighted-Residual-Connections (WRC)
Cross-Stage-Partial-connections (CSP)
Cross mini-Batch Normalization (CmBN)
Self-adversarial-training (SAT)
Mish-activation
Mosaic data augmentation
CmBN
DropBlock regularization
CIoU loss

本文的主要贡献如下：
1. 提出了一种高效而强大的目标检测模型。它使每个人都可以使用1080 Ti或2080 Ti GPU 训练超快速和准确的目标检测器。
2. 在检测器训练期间，验证了SOTA的Bag-of Freebies 和Bag-of-Specials方法的影响。
3. 改进了SOTA的方法，使它们更有效，更适合单GPU训练，包括CBN [89]，PAN [49]，SAM [85]等。文章将目前主流的目标检测器框架进行拆分：input、backbone、neck 和 head.

对于GPU，作者在卷积层中使用：CSPResNeXt50 / CSPDarknet53
对于VPU，作者使用分组卷积，但避免使用（SE）块-具体来说，它包括以下模型：EfficientNet-lite / MixNet / GhostNet / MobileNetV3

作者的目标是在输入网络分辨率，卷积层数，参数数量和层输出（filters）的数量之间找到最佳平衡。

总结一下YOLOv4框架：

Backbone：CSPDarknet53
Neck：SPP，PAN
Head：YOLOv3
YOLOv4 = CSPDarknet53+SPP+PAN+YOLOv3

其中YOLOv4用到相当多的技巧：

用于backbone的BoF：CutMix和Mosaic数据增强，DropBlock正则化，Class label smoothing
用于backbone的BoS：Mish激活函数，CSP，MiWRC
用于检测器的BoF：CIoU-loss，CmBN，DropBlock正则化，Mosaic数据增强，Self-Adversarial 训练，消除网格敏感性，对单个ground-truth使用多个anchor，Cosine annealing scheduler，最佳超参数，Random training shapes
用于检测器的Bos：Mish激活函数，SPP，SAM，PAN，DIoU-NMS。

**YOLOv5**
2020年2月YOLO之父Joseph Redmon宣布退出计算机视觉研究领域，2020 年 4 月 23 日YOLOv4 发布，2020 年 6 月 10 日YOLOv5发布。
他们公布的结果表明，YOLOv5 的表现要优于谷歌开源的目标检测框架 EfficientDet，尽管 YOLOv5 的开发者没有明确地将其与 YOLOv4 进行比较，但他们却声称 YOLOv5 能在 Tesla P100 上实现 140 FPS 的快速检测；相较而言，YOLOv4 的基准结果是在 50 FPS 速度下得到的
不仅如此，他们还提到「YOLOv5 的大小仅有 27 MB。」对比一下：使用 darknet 架构的 YOLOv4 有 244 MB。这说明 YOLOv5 实在特别小，比 YOLOv4 小近 90%。而在准确度指标上，「YOLOv5 与 YOLOv4 相当」。
因此总结起来，YOLOv5 宣称自己速度非常快，有非常轻量级的模型大小，同时在准确度方面又与 YOLOv4 基准相当。
大家对YOLOV5命名是争议很大，因为YOLOV5相对于YOLOV4来说创新性的地方很少。不过它的性能应该还是有的，现在kaggle上active检测的比赛小麦检测前面的选手大部分用的都是YOLOV5的框架。

##3、对两阶段目标检测模型 Faster-RCNN进行一个详细介绍
![Alt text](./3.1.png)
![Alt text](./3.2.png)
![Alt text](./3.3.png)
从上面的三张图可以看出，Faster R CNN由下面几部分组成：
1.数据集，image input
2.卷积层CNN等基础网络，提取特征得到feature map
3-1.RPN层，再在经过卷积层提取到的feature map上用一个3x3的slide window，去遍历整个feature map,在遍历过程中每个window中心按rate，scale（1:2,1:1,2:1）生成9个anchors，然后再利用全连接对每个anchors做二分类（是前景还是背景）和初步bbox regression，最后输出比较精确的300个ROIs。
3-2.把经过卷积层feature map用ROI pooling固定全连接层的输入维度。
4.然后把经过RPN输出的rois映射到ROIpooling的feature map上进行bbox回归和分类。

**SPP-Net**是出自论文《Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition》
由于一般的网络结构中都伴随全连接层，全连接层的参数就和输入图像大小有关，因为它要把输入的所有像素点连接起来,需要指定输入层神经元个数和输出层神经元个数，所以需要规定输入的feature的大小。而SPP-NET正好解决了这个问题。
![Alt text](./3.4.png)
总结而言，当网络输入的是一张任意大小的图片，这个时候我们可以一直进行卷积、池化，直到网络的倒数几层的时候，也就是我们即将与全连接层连接的时候，就要使用金字塔池化，使得任意大小的特征图都能够转换成固定大小的特征向量，这就是空间金字塔池化的意义（多尺度特征提取出固定大小的特征向量）。

**ROI pooling layer**实际上是SPP-NET的一个精简版，SPP-NET对每个proposal使用了不同大小的金字塔映射，而ROI pooling layer只需要下采样到一个7x7的特征图.对于VGG16网络conv5_3有512个特征图，这样所有region proposal对应了一个77512维度的特征向量作为全连接层的输入.

为什么要pooling成7×7的尺度？是为了能够共享权重。Faster RCNN除了用到VGG前几层的卷积之外，最后的全连接层也可以继续利用。当所有的RoIs都被pooling成（512×7×7）的feature map后，将它reshape 成一个一维的向量，就可以利用VGG16预训练的权重，初始化前两层全连接.

**Bbox 回归**
![Alt text](./3.5.png)
那么经过何种变换才能从图11中的窗口P变为窗口呢？比较简单的思路就是：
![Alt text](./3.6.png)
![Alt text](./3.7.png)

**RPN**
![Alt text](./3.8.png)
Feature Map进入RPN后，先经过一次33的卷积，同样，特征图大小依然是6040,数量512，这样做的目的应该是进一步集中特征信息，接着看到两个全卷积,即kernel_size=11,p=0,stride=1;
![Alt text](./3.9.png)
如上图中标识：
① rpn_cls：6040512-d ⊕ 1151218 > 604092 逐像素对其9个Anchor box进行二分类
② rpn_bbox：6040512-d ⊕ 1151236>60409*4 逐像素得到其9个Anchor box四个坐标信息

##4、对单阶段目标检测模型 Yolov5 进行一个详细介绍
YOLOv5网络结构图
![Alt text](./4.1.png)
上图即Yolov5的网络结构图，可以看出，还是分为输入端、Backbone、Neck、Prediction四个部分。
（1）输入端：Mosaic数据增强、自适应锚框计算
（2）Backbone：Focus结构，CSP结构
（3）Neck：FPN+PAN结构
（4）Prediction：GIOU_Loss

**输入端**
（1）Mosaic数据增强
Yolov5的输入端采用了和Yolov4一样的Mosaic数据增强的方式。
Mosaic数据增强提出的作者也是来自Yolov5团队的成员，不过，随机缩放、随机裁剪、随机排布的方式进行拼接，对于小目标的检测效果还是很不错的。
（2）自适应锚框计算
在Yolo算法中，针对不同的数据集，都会有初始设定长宽的锚框。
在网络训练中，网络在初始锚框的基础上输出预测框，进而和真实框groundtruth进行比对，计算两者差距，再反向更新，迭代网络参数。
在Yolov3、Yolov4中，训练不同的数据集时，计算初始锚框的值是通过单独的程序运行的。
但Yolov5中将此功能嵌入到代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。
（3）自适应图片缩放
在常用的目标检测算法中，不同的图片长宽都不相同，因此常用的方式是将原始图片统一缩放到一个标准尺寸，再送入检测网络中。
比如Yolo算法中常用416×416，608×608等尺寸，比如对下面800*600的图像进行变换。
但Yolov5代码中对此进行了改进，也是Yolov5推理速度能够很快的一个不错的trick。作者认为，在项目实际使用时，很多图片的长宽比不同。
因此缩放填充后，两端的黑边大小都不同，而如果填充的比较多，则存在信息冗余，影响推理速度。因此在Yolov5代码中datasets.py的letterbox函数中进行了修改，对原始图像自适应的添加最少的黑边。

**Backbone**
（1）	Focus结构
![Alt text](./4.2.png)
Focus结构，在Yolov3&Yolov4中并没有这个结构，其中比较关键是切片操作。
比如右图的切片示意图，4×4×3的图像切片后变成2×2×12的特征图。
以Yolov5s的结构为例，原始608×608×3的图像输入Focus结构，采用切片操作，先变成304×304×12的特征图，再经过一次32个卷积核的卷积操作，最终变成304×304×32的特征图。
需要注意的是：Yolov5s的Focus结构最后使用了32个卷积核，而其他三种结构，使用的数量有所增加
（2）	CSP结构
Yolov5与Yolov4不同点在于，Yolov4中只有主干网络使用了CSP结构，而Yolov5中设计了两种CSP结构，以Yolov5s网络为例，以CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中。

**Neck**
Yolov5现在的Neck和Yolov4中一样，都采用FPN+PAN的结构，Yolov5的Neck结构中，采用借鉴CSPNet设计的CSP2结构，加强网络特征融合的能力。

**输出端**
（1）、Bounding box损失函数
Yolov5中采用其中的GIOU_Loss做Bounding box的损失函数。
（2）、nms非极大值抑制
在目标检测的后处理过程中，针对很多目标框的筛选，通常需要nms操作。
Yolov4在DIOU_Loss的基础上采用DIOU_nms的方式，而Yolov5中仍然采用加权nms的方式。
