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

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting beautifulsoup4
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)
    [K     |████████████████████████████████| 122kB 17.3MB/s eta 0:00:01
    [?25hCollecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4)
      Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
    Installing collected packages: soupsieve, beautifulsoup4
    Successfully installed beautifulsoup4-4.9.3 soupsieve-2.2.1



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

YOLOv1
核心思想：将整张图片作为网络的输入（类似于Faster-RCNN），直接在输出层对BBox的位置和类别进行回归。
实现方法
•	将一幅图像分成SxS个网格(grid cell)，如果某个object的中心 落在这个网格中，则这个网格就负责预测这个object。

![](https://ai-studio-static-online.cdn.bcebos.com/7717e6282182499a89afb019fa3d5042d613f91168c94f67a0e3024ae9aede3f)

•	每个网络需要预测B个BBox的位置信息和confidence（置信度）信息，一个BBox对应着四个位置信息和一个confidence信息。confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息：
 
其中如果有object落在一个grid cell里，第一项取1，否则取0。 第二项是预测的bounding box和实际的groundtruth之间的IoU值。
•	每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5*B+C)的一个tensor。（注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。）
•	举例说明: 在PASCAL VOC中，图像输入为448x448，取S=7，B=2，一共有20个类别(C=20)。则输出就是7x7x30的一个tensor。整个网络结构如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/41879abfc0664a889594668abff0e383f65f3c30dac64a4797ed7cdf936ebd04)
•	在test的时候，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score:
 
等式左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息。
•	得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行NMS处理，就得到最终的检测结果。
简单的概括就是：
(1) 给个一个输入图像，首先将图像划分成7*7的网格
(2) 对于每个网格，我们都预测2个边框（包括每个边框是目标的置信度以及每个边框区域在多个类别上的概率）
(3) 根据上一步可以预测出7*7*2个目标窗口，然后根据阈值去除可能性比较低的目标窗口，最后NMS去除冗余窗口即可
损失函数
在实现中，最主要的就是怎么设计损失函数，让这个三个方面得到很好的平衡。作者简单粗暴的全部采用了sum-squared error loss来做这件事。
这种做法存在以下几个问题：
•	第一，8维的localization error和20维的classification error同等重要显然是不合理的；
•	第二，如果一个网格中没有object（一幅图中这种网格很多），那么就会将这些网格中的box的confidence push到0，相比于较少的有object的网格，这种做法是overpowering的，这会导致网络不稳定甚至发散。
解决办法：
•	更重视8维的坐标预测，给这些损失前面赋予更大的loss weight。
•	对没有object的box的confidence loss，赋予小的loss weight。
•	有object的box的confidence loss和类别的loss的loss weight正常取1。
对不同大小的box预测中，相比于大box预测偏一点，小box预测偏一点肯定更不能被忍受的。而sum-square error loss中对同样的偏移loss是一样。
为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。这个参考下面的图很容易理解，小box的横轴值较小，发生偏移时，反应到y轴上相比大box要大。（也是个近似逼近方式）
![](https://ai-studio-static-online.cdn.bcebos.com/8313fb482f3044c28e9fa685ea1a9501457be03b83d341f3a61c7039f659d39b)
一个网格预测多个box，希望的是每个box predictor专门负责预测某个object。具体做法就是看当前预测的box与ground truth box中哪个IoU大，就负责哪个。这种做法称作box predictor的specialization。
最后整个的损失函数如下所示：
![](https://ai-studio-static-online.cdn.bcebos.com/62c5a3e2722f4ad986481e0d81a5370e5395ed242a434537a4277225b03131a1)
这个损失函数中：
•	只有当某个网格中有object的时候才对classification error进行惩罚。
•	只有当某个box predictor对某个ground truth box负责的时候，才会对box的coordinate error进行惩罚，而对哪个ground truth box负责就看其预测值和ground truth box的IoU是不是在那个cell的所有box中最大。
其他细节，例如使用激活函数使用leak RELU，模型用ImageNet预训练等等
优点
•	快速，pipline简单.
•	背景误检率低。
•	通用性强。YOLO对于艺术类作品中的物体检测同样适用。它对非自然图像物体的检测率远远高于DPM和RCNN系列检测方法。
缺点
•	由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。
•	虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。
•	YOLO loss函数中，大物体IOU误差和小物体IOU误差对网络训练中loss贡献值接近（虽然采用求平方根方式，但没有根本解决问题）。因此，对于小物体，小的IOU误差也会对网络优化过程造成很大的影响，从而降低了物体检测的定位准确性。
YOLOv2（YOLO9000）
YOLOv2相对v1版本，在继续保持处理速度的基础上，从预测更准确（Better），速度更快（Faster），识别对象更多（Stronger）这三个方面进行了改进。其中识别更多对象也就是扩展到能够检测9000种不同对象，称之为YOLO9000。
YOLOv3
OLO v3的模型比之前的模型复杂了不少，可以通过改变模型结构的大小来权衡速度与精度。
速度对比如下：
![](https://ai-studio-static-online.cdn.bcebos.com/27468f79b7c24b8289baa5f01de2d482a8845650bd0246f1b1cabae5ff545b5a)
简而言之，YOLOv3 的先验检测（Prior detection）系统将分类器或定位器重新用于执行检测任务。他们将模型应用于图像的多个位置和尺度。而那些评分较高的区域就可以视为检测结果。此外，相对于其它目标检测方法，我们使用了完全不同的方法。我们将一个单神经网络应用于整张图像，该网络将图像划分为不同的区域，因而预测每一块区域的边界框和概率，这些边界框会通过预测的概率加权。我们的模型相比于基于分类器的系统有一些优势。它在测试时会查看整个图像，所以它的预测利用了图像中的全局信息。与需要数千张单一目标图像的 R-CNN 不同，它通过单一网络评估进行预测。这令 YOLOv3 非常快，一般它比 R-CNN 快 1000 倍、比 Fast R-CNN 快 100 倍。
改进之处
•	多尺度预测 （引入FPN）。
•	更好的基础分类网络（darknet-53, 类似于ResNet引入残差结构）。
•	分类器不在使用Softmax，分类损失采用binary cross-entropy loss（二分类交叉损失熵）
YOLOv3不使用Softmax对每个框进行分类，主要考虑因素有两个：
1.	Softmax使得每个框分配一个类别（score最大的一个），而对于Open Images这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类。
2.	Softmax可被独立的多个logistic分类器替代，且准确率不会下降。
分类损失采用binary cross-entropy loss。
多尺度预测
每种尺度预测3个box, anchor的设计方式仍然使用聚类,得到9个聚类中心,将其按照大小均分给3个尺度.
•	尺度1: 在基础网络之后添加一些卷积层再输出box信息.
•	尺度2: 从尺度1中的倒数第二层的卷积层上采样(x2)再与最后一个16x16大小的特征图相加,再次通过多个卷积后输出box信息.相比尺度1变大两倍.
•	尺度3: 与尺度2类似,使用了32x32大小的特征图.
参见网络结构定义文件 yolov3.cfg
https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
基础网络 Darknet-53
![](https://ai-studio-static-online.cdn.bcebos.com/85318177d616445fa97df9fc5adbab2c94aace0fc6be46db822071639b3da4e7)
darknet-53
仿ResNet, 与ResNet-101或ResNet-152准确率接近,但速度更快.对比如下:
![](https://ai-studio-static-online.cdn.bcebos.com/1cceac3f9ec2497c8b70869140fda883375f037f8ea74798acc320df6dab373b)
主干架构的性能对比
检测结构如下：
![](https://ai-studio-static-online.cdn.bcebos.com/5a27bdf752df43fa982bc435be410213b7268a8a9ba64463b055252a311f28a9)
![](https://ai-studio-static-online.cdn.bcebos.com/a17dd3f57b8e46c187d51bb07b659a762c50ae66bee34033884347f81058d2e5)
YOLOv3在mAP@0.5及小目标APs上具有不错的结果,但随着IOU的增大,性能下降,说明YOLOv3不能很好地与ground truth切合.边框预测![](https://ai-studio-static-online.cdn.bcebos.com/a6782f35dd334a9689f226c293bd157c8a84e358051a4ad28b89420e51693111)
图 2：带有维度先验和定位预测的边界框。我们边界框的宽和高以作为离聚类中心的位移，并使用 Sigmoid 函数预测边界框相对于滤波器应用位置的中心坐标。
仍采用之前的lYOLOv4
YOLOv4: Optimal Speed and Accuracy of Object Detection

论文：https://arxiv.org/abs/2004.10934

代码：https://github.com/AlexeyAB/darknet

YOLOv4！

![](https://ai-studio-static-online.cdn.bcebos.com/1a29917b809f4dac9a983f27752d9478b27445c3a6c341eb83f4ffbc89aae799)ogis，其中cx,cy是网格的坐标偏移量,pw,ph是预设的anchor box的边长.最终得到的边框坐标值是b*,而网络学习目标是t*，用sigmod函数、指数转换。
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

1. 提出了一种高效而强大的目标检测模型。它使每个人都可以使用1080 Ti或2080 Ti GPU 训练超快速和准确的目标检测器（牛逼！）。

2. 在检测器训练期间，验证了SOTA的Bag-of Freebies 和Bag-of-Specials方法的影响。

3. 改进了SOTA的方法，使它们更有效，更适合单GPU训练，包括CBN [89]，PAN [49]，SAM [85]等。文章将目前主流的目标检测器框架进行拆分：input、backbone、neck 和 head.

具体如下图所示：![](https://ai-studio-static-online.cdn.bcebos.com/45c6a65b7d5349239ddd14fb27b1f609578dbebfee054669b3ff045df60d0bca)
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
用于检测器的Bos：Mish激活函数，SPP，SAM，PAN，DIoU-NMS
看看YOLOv4部分组件：![](https://ai-studio-static-online.cdn.bcebos.com/7cfd1713322f46628f59026616e5d838dc3396d2eeaf4f5aaeab097639e3d204)
YOLOv5：
2020年2月YOLO之父Joseph Redmon宣布退出计算机视觉研究领域，2020 年 4 月 23 日YOLOv4 发布，2020 年 6 月 10 日YOLOv5发布。

YOLOv5源代码：

https://github.com/ultralytics/yolov5

他们公布的结果表明，YOLOv5 的表现要优于谷歌开源的目标检测框架 EfficientDet，尽管 YOLOv5 的开发者没有明确地将其与 YOLOv4 进行比较，但他们却声称 YOLOv5 能在 Tesla P100 上实现 140 FPS 的快速检测；相较而言，YOLOv4 的基准结果是在 50 FPS 速度下得到的，参阅：https://blog.roboflow.ai/yolov5![](https://ai-studio-static-online.cdn.bcebos.com/fe49f83dd6a649f3a3f0d867d5bc04d6019daa8c3a254683a9ae6f304356a6ac)


