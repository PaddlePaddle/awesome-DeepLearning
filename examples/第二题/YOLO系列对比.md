# **1. 目标检测任务解决正负样本不均衡的手段**

**1. 1. focal loss交叉熵**

根据置信度结果动态调整交叉熵损失解决类别不平衡问题（当预测正确的置信度增加时，loss的权重系数会逐渐衰减至0，这样模型训练的loss更关注难例，而大量容易的例子其loss贡献很低）

原有的二分类交叉熵：

![](https://ai-studio-static-online.cdn.bcebos.com/6a8ea3fdc3ef41e6be2a3edd3a69c67268ef06de68574f35bab19bcbac3418cb)


普通的交叉熵损失对于负样本而言，在输出概率较小时，损失还不够小（同理对于正样本而言，在输出概率较大时，损失也还不够小）。这样的损失函数在大量简单样本的迭代过程中比较缓慢且可能无法优化至最优。

Focal loss对此做了改进，其损失函数如下：

![](https://ai-studio-static-online.cdn.bcebos.com/f58d7eefe13b4b06b4de532b001270bf0a488274e2c44686808a94be5738c01e)
![](https://ai-studio-static-online.cdn.bcebos.com/24ae8ca277924a84971b2cdc0149042f5eca444799414fb7b6bef269bf831e5b)


1. 在原有的基础上加了一个gamma因子，用于解决难易样本不平衡问题，其中gamma>0使得易分类样本的损失进一步降低，让模型训练更关注于困难的、错分的样本。

2. 加入平衡因子alpha，用来平衡正负样本本身的比例不均：文中alpha取0.25，即正样本要比负样本占比小，这是因为负例易分。

3. 只添加alpha虽然可以平衡正负样本的重要性，但是无法解决简单与困难样本的问题。

4. gamma调节简单样本损失降低的速率，当gamma为0时即为交叉熵损失函数，当gamma增加时，调整因子的影响也在增加。实验发现gamma为2是最优。

5. 模型的专注度：正难 > 负难 > 正易 > 负易。

Focal loss的优缺点：

优点：
1. 解决了one-stage object detection中图片中正负样本（前景和背景）不均衡的问题；
2. 降低简单样本的权重，使损失函数更关注困难样本；

缺点：
1. 模型很容易收到噪声干扰：会将噪声当成复杂样本，使模型过拟合退化；
2. 模型的初期，数量多的一类可能主导整个loss，所以训练初期可能训练不稳定；
两个参数αt和γ具体的值很难定义，需要自己调参，调的不好可能效果会更差。

**1. 2. Fast R-CNN**

两阶段检测模型中，提出的RoI Proposal在输入R-CNN子网络前，我们有机会对正负样本（背景类和前景类）的比例进行调整。

通常，背景类的RoI Proposal个数要远远多于前景类，Fast R-CNN的处理方式是随机对两种样本进行上采样和下采样，以使每一batch的正负样本比例保持在1:3，这一做法缓解了类别比例不均衡的问题，是两阶段方法相比单阶段方法具有优势的地方，也被后来的大多数工作沿用。

**1. 3. GHM(Gradient Harmonizing Mechanism)**

正负样本不均衡问题一直是One-stage目标检测中被大家所诟病的地方，He Keming等人提出了Focal Loss来解决这个问题。而AAAI2019上的一篇论文《Gradient Harmonized Single-stage Detector》则尝试从梯度分部的角度，来解释样本分步不均衡给目one-stage目标检测带来的瓶颈本质，并尝试提出了一种新的损失函数：GHM(Gradient Harmonizing Mechanism)来解决这个问题。

GHM-C和GHM-R是定义的损失函数，因此可以非常方便的安插到很多目标检测方法中，作者以focal loss（我猜测应该是以RetinaNet作为baseline），对交叉熵，focal loss和GHM-C做了对比，发现GHM-C在focal loss 的基础上在AP上提升了0.2个百分点：

![](https://ai-studio-static-online.cdn.bcebos.com/f3adfb0a0d7f47fcb770b0924b4a9942a1adbfeb7b634207a8908c315509ec8b)


如果用GHM-R代替two-stage detector中的smooth L1，AP上又会有提升：

![](https://ai-studio-static-online.cdn.bcebos.com/f06b469a51a5403bb81e32d174538c00def133315e6e4694a15a8cf4984dc114)


如果用上GHM-C和GHM-R，准确率的提升很明显，大概有2个百分点：


![](https://ai-studio-static-online.cdn.bcebos.com/5b59849cf4834f3fb4712768e434d5a70fd50e62830b4b00b7422f1d08b9edab)


# 2. **yolo系列的对比**

* YOLO，即You Only Look Once的缩写，是一个基于卷积神经网络的物体检测算法。
* YOLO系列算法是一类典型的one-stage目标检测算法，其利用anchor box将分类与目标定位的回归问题结合起来，从而做到了高效、灵活和泛化性能好。其backbone网络darknet也可以替换为很多其他的框架，所以在工程领域也十分受欢迎。下面我将从yolov1出发，在网络结构模型角度对yolo系列进行对比。

**2. 1. yolov1**

![](https://ai-studio-static-online.cdn.bcebos.com/a8033fa830434af99da3d8ee15ff8630c6b5b80d6a8e4954bd2b724abef3fddd)


![](https://ai-studio-static-online.cdn.bcebos.com/bbe47b6e70f24d8d83ea35bc1df799b2f459b560147d4cc1bbffa3cfe9dd46ee)


YOLO网络借鉴了GoogLeNet分类网络结构。不同的是，yolo未使用inception module，而是使用1x1卷积层（此处1x1卷积层的存在是为了跨通道信息整合）+3x3卷积层简单替代。 由24个卷积层与2个全连接层构成，网络入口为448x448(v2为416x416)，图片进入网络先经过resize，YOLO网络最终的全连接层的输出结果为一个张量，其维度为$\left(S*S\right)\left(B*5+C\right)$

**2. 2.  yolov2**

![](https://ai-studio-static-online.cdn.bcebos.com/62f7160caff54ff8a24b0e65b9befee62dc12034623a430cbe55d3db07ac4363)



![](https://ai-studio-static-online.cdn.bcebos.com/a206e7be38044c328d4530701e90a9aa0fc98a50a98e4395818770e84a49fddc)


相对于yolov1，yolov2在网络结构上进行了如下改进：
1. 网络采用DarkNet-19 主干网络的升级。
2. Batch Normalization： v1中也大量用了Batch Normalization，同时在定位层后边用了dropout，v2中取消了dropout，在卷积层全部使用Batch Normalization。BN能够给模型收敛带来显著地提升，同时也消除了其他形式正则化的必要。
3. 使用anchors：借鉴faster R-CNN和SSD，对于一个中心点，使用多个anchor，得到多个bounding box，每个bounding box包含4个位置坐标参数(x,y,w,h)和21个类别概率信息。而在yolov1中，每个grid（对应anchor），仅预测一次类别，而且只有两个bounding box来进行坐标预测。v1中直接在卷积层之后使用全连接层预测bbox的坐标。v2借鉴Faster R-CNN的思想预测bbox的偏移.移除了全连接层,并且删掉了一个pooling层使特征的分辨率更大一些。v1中每张图片预测7x7x2=98个box,而v2加上Anchor Boxes能预测13X13X5(5+20)个box。
4. 去掉全连接层：和SSD一样，模型中只包含卷积和平均池化层（平均池化是为了变为一维向量，做softmax分类）。这样做一方面是由于物体检测中的目标，只是图片中的一个区块，它是局部感受野，没必要做全连接。而是为了输入不同尺寸的图片，如果采用全连接，则只能输入固定大小图片了。
5. Yolov2的损失函数跟yolov1差别不大，唯一的差别就是关于bbox的w和h的损失去掉了根号，即：

![](https://ai-studio-static-online.cdn.bcebos.com/5985c02e1b3b46eb951e37e5724a8caf8a97dae8816c4069bcadf361f3b6e027)


**2. 3.  yolov3**

![](https://ai-studio-static-online.cdn.bcebos.com/18bb48357ff244bebfa31cefcc5e601b4a37d2d0113b43d8ae5f1211d0e74a98)



![](https://ai-studio-static-online.cdn.bcebos.com/28b15882f3ca4a14b71b50c1a1c9016f1e41562db94f4de98f7a3ffd175e162b)


相对于yolov2，yolov3在网络结构上进行了如下改进：
1. 引入FPN （多尺度预测）
YOLOv3 借鉴了FPN的思想，从不同尺度提取特征。相比yolov2，yolov3 提取最后3层特征图，不仅在每个特征图上分别独立做预测，同时通过将小特征图上采样到与大的特征图相同大小，然后与大的特征图拼接做进一步预测。用维度聚类的思想聚类出 9 种尺度的 anchor box，将 9 种尺度的 anchor box 均匀的分配给3种尺度的特征图。
2. 对Darknet-53主干网络进行了改进。
3. softmax被替代：在实际应用场合中，一个物体有可能输入多个类别，单纯的单标签分类在实际场景中存在一定的限制。因此在yolov3 在网络结构中把原先的softmax层换成了逻辑回归层，从而实现把单标签分类改成多标签分类。用多个logistic分类器代替 softmax 并不会降低准确率，可以维持yolo的检测精度不下降。
4. 置信损失用的是sigmoid函数代替softmax来计算概率分布，损失函数采用采用的是BCE。

**2. 4. yolov4**

![](https://ai-studio-static-online.cdn.bcebos.com/e6b69d15d7d3461da4c40dafcc668ffcdbe32bd608fb4cf39d36b8677e818da3)


![](https://ai-studio-static-online.cdn.bcebos.com/27441b8570d04cb6a2681afe851d5a2d3f7690fdf1914833b61575b295cc3856)



相对于yolov3，yolov4在网络结构上进行了如下改进：
1. 主干网络改进：CSPDarknet53作为Backbone。
2. 训练方法改进：数据增强把四张图片拼成一张图片，等价于增大了mini-batch。还有自对抗训练数据增强方法。这是在一张图上，让神经网络反向更新图像，对图像做改变扰动，然后在这个图像上训练。
3. 吸收了一些近年来最新深度学习网络的技巧。如CutMix数据增强，Swish、Mish激活函数。
4. CmBN 跨最小批的归一化（Cross mini-batch Normal），在CBN的基础上改进；CmBN吸收了BN和CBN的特点。
5. 使用IOU损失代替MSE损失。

**2. 5.  yolov5**

![](https://ai-studio-static-online.cdn.bcebos.com/00032bea0d974c7994e8c5999c4d4ada62f435fefee24ebab840d819155428a4)



![](https://ai-studio-static-online.cdn.bcebos.com/7ce26cb8816342fab7067f40e841785121da33a062e5482488cae30d1453dd6f)


相对于yolov4，yolov5在网络结构上进行了如下改进：
1. 输入端：Mosaic数据增强、自适应锚框计算：yolov5会进行三种数据增强：缩放，色彩空间调整和马赛克增强。其中马赛克增强是通过将四张图像进行随机缩放、随机裁剪、随机分布方式进行拼接，小目标的检测效果得到提升。yolov5还将初始化anchor的功能嵌入到代码中，每次训练数据集之前，都会自动计算该数据集最合适的Anchor尺寸，该功能可以在代码中设置超参数进行关闭。
2. Backbone：Focus结构，CSP结构：yolov5中设计了两种CSP结构，CSP1_X应用于BackBone主干网络，另一种CSP_2X结构则应用于Neck中。在yolov5中，原始的图像输入Focus结构，采用切片操作，先变成的特征图，再经过一次32个卷积核的卷积操作，最终变成的特征图，focus结构减少了FLOPs，提高了速度，但对于模型的精度mAP没有提升。
3. Neck：采用FPN+PAN结构：yolov5的Neck结构中，采用借鉴CSPnet设计的CSP2_X结构，加强网络特征融合的能力。
4. Prediction：损失函数采用GIOU_Loss。


