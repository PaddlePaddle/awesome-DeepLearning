# Faster-RCNN 模型介绍

最新的物体检测网络依赖于候选框(生成)算法来假设物体位置。最新的进展如SPPnet和Fast R-CNN已经减少了检测网络的时间，(间接)凸显出候选框计算成为算法时间的瓶颈。Faster-RCNN引入了Region Proposal Network (RPN) ，它和检测网络共享整图的卷积特征，这样使得候选框的计算几乎不额外占用时间。RPN是一个全卷积网络，可同时预测物体外接框和每个位置是否为物体的得分。RPN采用端到端的方式进行训练，产生高质量的候选框，进而被Fast R-CNN用来做检测。Faster-RCNN通过共享卷积特征，进一步融合RPN和Fast R-CNN为一个网络——使用最近流行的基于注意力机制的网络技术，RPN单元指引统一后的网络查看的地方。


# Faster-RCNN的模型结构

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/519d408eb5ab4857ae6ec524b46646defdd14061e78d439d8af2c1741e17422e" width="400" hegiht="" ></center>


Faster-RCNN有两部分组成：RPN和Fast-RCNN。两者共享同一个backbone。具体来说，Faster-RCNN由以下几部分组成：

1、backbone（VGG，ResNet等）

2、neck（FPN，原版的faster没有，FPN出来之后后人才加上的）

3、rpn_head（RPNHead）

4、bbox_roi_extractor（SingleRoIExtractor > RoIAlign，RoIPool）

5、bbox_head（SharedFCBBoxHead）

# 算法步骤

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ad47e4b503a9496e8fcef0179a4ba142085f7c6779ce435da2d96078e213363e" width="400" hegiht="" ></center>

１．Conv layers.作为一种cnn网络目标检测的方法，faster_rcnn首先使用一组基础conv+relu+pooling层提取image的feture map。该feature map被共享用于后续的RPN层和全连接层。

２．Region Proposal Networks.RPN层是faster-rcnn最大的亮点，RPN网络用于生成region proposcals.该层通过softmax判断anchors属于foreground或者background，再利用box regression修正anchors获得精确的propocals（anchors也是作者自己提出来的，后面我们会认真讲）

３．Roi Pooling.该层收集输入的feature map 和 proposcal，综合这些信息提取proposal feature map，送入后续的全连接层判定目标类别。

４．Classification。利用proposal feature map计算proposcal类别，同时再次bounding box regression获得检验框的最终精确地位置




# RCNN系列中 Faster-RCNN的特点

### R-CNN:

(1)image input；

(2)利用selective search 算法在图像中从上到下提取2000个左右的Region Proposal；

(3)将每个Region Proposal缩放(warp)成227*227的大小并输入到CNN，将CNN的fc7层的输出作为特征；

(4)将每个Region Proposal提取的CNN特征输入到SVM进行分类；

(5)对于SVM分好类的Region Proposal做边框回归，用Bounding box回归值校正原来的建议窗口，生成预测窗口坐标.

### 缺陷:

(1) 训练分为多个阶段，步骤繁琐：微调网络+训练SVM+训练边框回归器；

(2) 训练耗时，占用磁盘空间大；5000张图像产生几百G的特征文件；

(3) 速度慢：使用GPU，VGG16模型处理一张图像需要47s；

(4) 测试速度慢：每个候选区域需要运行整个前向CNN计算；

(5) SVM和回归是事后操作，在SVM和回归过程中CNN特征没有被学习更新.

### FAST-RCNN:

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5aa117d1c95d4aaba357fc2d1cffe2aef1defda7f8214595ba4d70bdc3459aef" width="400" hegiht="" ></center>


(1)image input；

(2)利用selective search 算法在图像中从上到下提取2000个左右的建议窗口(Region Proposal)；

(3)将整张图片输入CNN，进行特征提取；

(4)把建议窗口映射到CNN的最后一层卷积feature map上；

(5)通过RoI pooling层使每个建议窗口生成固定尺寸的feature map；

(6)利用Softmax Loss(探测分类概率) 和Smooth L1 Loss(探测边框回归)对分类概率和边框回归(Bounding box regression)联合训练.

### 相比R-CNN，主要两处不同:

(1)最后一层卷积层后加了一个ROI pooling layer；

(2)损失函数使用了多任务损失函数(multi-task loss)，将边框回归直接加入到CNN网络中训练
改进:

(1) 测试时速度慢：R-CNN把一张图像分解成大量的建议框，每个建议框拉伸形成的图像都会单独通过CNN提取特征.实际上这些建议框之间大量重叠，特征值之间完全可以共享，造成了运算能力的浪费.
FAST-RCNN将整张图像归一化后直接送入CNN，在最后的卷积层输出的feature map上，加入建议框信息，使得在此之前的CNN运算得以共享.

(2) 训练时速度慢：R-CNN在训练时，是在采用SVM分类之前，把通过CNN提取的特征存储在硬盘上.这种方法造成了训练性能低下，因为在硬盘上大量的读写数据会造成训练速度缓慢.
FAST-RCNN在训练时，只需要将一张图像送入网络，每张图像一次性地提取CNN特征和建议区域，训练数据在GPU内存里直接进Loss层，这样候选区域的前几层特征不需要再重复计算且不再需要把大量数据存储在硬盘上.

(3) 训练所需空间大：R-CNN中独立的SVM分类器和回归器需要大量特征作为训练样本，需要大量的硬盘空间.FAST-RCNN把类别判断和位置回归统一用深度网络实现，不再需要额外存储.

(4) 由于ROI pooling的提出，不需要再input进行Corp和wrap操作，避免像素的损失，巧妙解决了尺度缩放的问题.

### FASTER -RCNN:

(1)输入测试图像；

(2)将整张图片输入CNN，进行特征提取；

(3)用RPN先生成一堆Anchor box，对其进行裁剪过滤后通过softmax判断anchors属于前景(foreground)或者后景(background)，即是物体or不是物体，所以这是一个二分类；同时，另一分支bounding box regression修正anchor box，形成较精确的proposal（注：这里的较精确是相对于后面全连接层的再一次box regression而言）

(4)把建议窗口映射到CNN的最后一层卷积feature map上；

(5)通过RoI pooling层使每个RoI生成固定尺寸的feature map；

(6)利用Softmax Loss(探测分类概率) 和Smooth L1 Loss(探测边框回归)对分类概率和边框回归(Bounding box regression)联合训练.

### 相比FASTER-RCNN，主要两处不同:

(1)使用RPN(Region Proposal Network)代替原来的Selective Search方法产生建议窗口；

(2)产生建议窗口的CNN和目标检测的CNN共享

### 改进: 如何高效快速产生建议框？

FASTER-RCNN创造性地采用卷积网络自行产生建议框，并且和目标检测网络共享卷积网络，使得建议框数目从原有的约2000个减少为300个，且建议框的质量也有本质的提高.

# 模型指标

Faster R-CNN是作者Ross Girshick继Fast R-CNN后的又一力作。同样使用VGG16作 为网络的backbone，推理速度在GPU上达到5fps(包括候选区域的生成)，准确率 也有进一步的提升。在2015年的ILSVRC以及COCO竞赛中获得多个项目的第一名。
