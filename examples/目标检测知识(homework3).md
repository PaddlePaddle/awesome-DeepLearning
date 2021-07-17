目标检测知识:

1目标检测任务解决正负样本不均衡问题的手段有哪些?

机器学习中，解决样本不均衡问题主要有2种思路：数据角度和算法角度。从数据角度出发，有扩大数据集、数据类别均衡采样等方法。接下来我们将重点关注算法层面的一些解决思路。：

解决

解决这一问题的基本思路是让正负样本在训练过程中拥有相同的话语权，比如利用采样与加权等方法。为了方便起见，我们把数据集中样本较多的那一类称为“大众类”，样本较少的那一类称为“小众类”。

采样

采样方法是通过对训练集进行处理使其从不平衡的数据集变成平衡的数据集，在大部分情况下会对最终的结果带来提升。

采样分为上采样（Oversampling）和下采样（Undersampling），上采样是把小众类复制多份，下采样是从大众类中剔除一些样本，或者说只从大众类中选取部分样本。

随机采样最大的优点是简单，但缺点也很明显。上采样后的数据集中会反复出现一些样本，训练出来的模型会有一定的过拟合；而下采样的缺点显而易见，那就是最终的训练集丢失了数据，模型只学到了总体模式的一部分。

上采样会把小众样本复制多份，一个点会在高维空间中反复出现，这会导致一个问题，那就是运气好就能分对很多点，否则分错很多点。为了解决这一问题，可以在每次生成新数据点时加入轻微的随机扰动，经验表明这种做法非常有效。

因为下采样会丢失信息，如何减少信息的损失呢？第一种方法叫做EasyEnsemble，利用模型融合的方法（Ensemble）：多次下采样（放回采样，这样产生的训练集才相互独立）产生多个不同的训练集，进而训练多个不同的分类器，通过组合多个分类器的结果得到最终的结果。第二种方法叫做BalanceCascade，利用增量训练的思想（Boosting）：先通过一次下采样产生训练集，训练一个分类器，对于那些分类正确的大众样本不放回，然后对这个更小的大众样本下采样产生训练集，训练第二个分类器，以此类推，最终组合所有分类器的结果得到最终结果。第三种方法是利用KNN试图挑选那些最具代表性的大众样本，叫做NearMiss，这类方法计算量很大，感兴趣的可以参考“Learning from Imbalanced Data”这篇综述的3.2.1节。

数据合成

数据合成方法是利用已有样本生成更多样本，这类方法在小数据场景下有很多成功案例，比如医学图像分析等。

![img](https://img-blog.csdn.net/20170915161326166?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVtaWxh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 

SMOTE为每个小众样本合成相同数量的新样本，这带来一些潜在的问题：一方面是增加了类之间重叠的可能性，另一方面是生成一些没有提供有益信息的样本。为了解决这个问题，出现两种方法：Borderline-SMOTE与ADASYN。

Borderline-SMOTE的解决思路是寻找那些应该为之合成新样本的小众样本。即为每个小众样本计算K近邻，只为那些K近邻中有一半以上大众样本的小众样本生成新样本。直观地讲，只为那些周围大部分是大众样本的小众样本生成新样本，因为这些样本往往是边界样本。确定了为哪些小众样本生成新样本后再利用SMOTE生成新样本。

![img](https://img-blog.csdn.net/20170915161350193?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmVtaWxh/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

 

横向是真实分类情况，纵向是预测分类情况，C(i,j)是把真实类别为j的样本预测为i时的损失，我们需要根据实际情况来设定它的值。

这种方法的难点在于设置合理的权重，实际应用中一般让各个分类间的加权损失值近似相等。当然这并不是通用法则，还是需要具体问题具体分析。

一分类

对于正负样本极不平衡的场景，我们可以换一个完全不同的角度来看待问题：把它看做一分类（One Class Learning）或异常检测（Novelty Detection）问题。这类方法的重点不在于捕捉类间的差别，而是为其中一类进行建模，经典的工作包括One-class SVM等。

说明：对于正负样本极不均匀的问题，使用异常检测，或者一分类问题，也是一个思路。

2yolov1、yolov2是在yolov3之前提出的目标检测网络、yolov4、yolov5是继yolov3之后提出的用于目标检测的网络，请对比yolov1、yolov2、yolov3、yolov4、yolov5五个模型、

YOLOv3和YOLOv2区别：

锚框使用Kmeans聚类的方法，一共九个锚框，每个尺寸的特征图呢使用3个锚框。

loss后面三项用二分类交叉熵代替原始的误差和平方。

多尺度预测，增加对细粒度物体的检测力度。

大量使用残差网络，对训练更深度的神经网络而言，可以消除梯度消失或梯度爆炸的问题。

在相同速度时，yolov5优于yolov3，yolov5有四种模型，可以灵活选择。

通过yolov3、yolov4和yolov5对比，可以得出在整体性能方面yolov4最优，但yolov5灵活性较强，具有四种网络模型，可以根据需求选择适当的模型。Yolov4整体优于yolov3。在相同速度时yolov5优于yolov3。

3对两阶段目标检测模型Faster-RCNN进行一个详细的介绍

Faster RCNN其实可以分为4个主要内容：

Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。

Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

Conv layers

Conv layers包含了conv，pooling，relu三种层。以python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构为例，Conv layers部分共有13个conv层，13个relu层，4个pooling层。这里有一个非常容易被忽略但是又无比重要的信息，在Conv layers中：

所有的conv层都是：kernel_size=3，pad=1，stride=1

所有的pooling层都是：kernel_size=2，pad=0，stride=2

Region Proposal Networks(RPN)

经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如R-CNN使用SS(Selective Search)方法生成检测框。而Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster R-CNN的巨大优势，能极大提升检测框的生成速度。

RoI pooling

而RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。

1. 原始的feature maps
2. RPN输出的proposal boxes（大小各不相同）

Classification

Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。

4对单阶段目标检测模型yolov5进行一个详细的介绍

网络结构:

![img](https://img-blog.csdnimg.cn/20210422080951390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

整体的大结构没有改变。

输入端：Mosaic数据增强、自适应锚框计算

Backbone：Focus结构，CSP结构

Neck：FPN+PAN结构

Prediction：GIOU_Loss

### 1、输入端

#### (1)数据增强

Yolov5的输入端采用了和Yolov4一样的Mosaic数据增强的方式。Mosaic数据增强提出的作者也是来自Yolov5团队的成员，不过，随机缩放、随机裁剪、随机排布的方式进行拼接，对于小目标的检测效果还是很不错的。

#### (2)自适应锚框计算

在Yolo算法中，针对不同的数据集，都会有初始设定长宽的锚框。在网络训练中，网络在初始锚框的基础上输出预测框，进而和真实框groundtruth进行比对，计算两者差距，再反向更新，迭代网络参数。因此初始锚框也是比较重要的一部分。
在Yolov3、Yolov4中，训练不同的数据集时，计算初始锚框的值是通过单独的程序运行的。但Yolov5中将此功能嵌入到代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。当然，如果觉得计算的锚框效果不是很好，也可以在代码中将自动计算锚框功能关闭。

#### (3)自适应图片缩放

在常用的目标检测算法中，不同的图片长宽都不相同，因此常用的方式是将原始图片统一缩放到一个标准尺寸，再送入检测网络中。比如Yolo算法中常用416×416，608×608等尺寸，比如对下面800*600的图像进行变换。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422081002213.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

但Yolov5代码中对此进行了改进，也是Yolov5推理速度能够很快的一个不错的trick。**作者认为,在项目实际使用时，很多图片的长宽比不同。因此缩放填充后，两端的黑边大小都不同，而如果填充的比较多，则存在信息冗余，影响推理速度。** 因此在Yolov5代码中datasets.py的**letterbox函数**中进行了修改，对原始图像自适应的添加最少的黑边。

![img](https://img-blog.csdnimg.cn/2021042208101142.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

图像高度上两端的黑边变少了，在推理时，计算量也会减少，即目标检测速度会得到提升。在YOLOv3讨论中，通过这种简单的改进，推理速度得到了37%的提升，可以说效果很明显。那究竟是怎么做的呢？

##### (1) 计算缩放比例

![img](https://img-blog.csdnimg.cn/20210422081020785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

原始缩放尺寸是416×416，都除以原始图像的尺寸后，可以得到0.52，和0.69两个缩放系数，选择小的缩放系数0.52。

##### (2) 计算缩放后的尺寸

![img](https://img-blog.csdnimg.cn/2021042208103046.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

原始图片的长宽都乘以最小的缩放系数0.52，宽变成了416，而高变成了312。

##### (3) 计算黑边填充数值

![img](https://img-blog.csdnimg.cn/20210422081040282.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

将416-312=104，得到原本需要填充的高度。再采用numpy中np.mod取余数的方式，得到8个像素，再除以2，即得到图片高度两端需要填充的数值。

**此外，需要注意的是：**
(1).这里填充的是黑色，即（0，0，0），而Yolov5中填充的是灰色，即（114,114,114），都是一样的效果。
(2).**训练**时没有采用缩减黑边的方式，还是**采用传统填充的方式**，即缩放到416×416大小。
只是在测试，**使用模型推理时，才采用缩减黑边的方式，** 提高目标检测，推理的速度。
(3).为什么np.mod函数的后面用32？因为Yolov5的网络经过5次下采样，而2的5次方，等于32。所以至少要去掉32的倍数，再进行取余。

### 2、backbone

#### (1)Focus结构

![img](https://img-blog.csdnimg.cn/20210422081052101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

**Focus结构**， 在Yolov3&Yolov4中并没有这个结构，其中比较关键是 **切片操作。** 比如右图的切片示意图，4×4×3的图像切片后变成2×2×12的特征图。以Yolov5s的结构为例，原始608×608×3的图像输入Focus结构，采用切片操作，先变成304×304×12的特征图，再经过一次32个卷积核的卷积操作，最终变成304×304×32的特征图。**需要注意的是**:Yolov5s的Focus结构最后使用了32个卷积核，而其他三种结构，使用的数量有所增加，先注意下，后面会讲解到**四种结构的不同点**。

#### (2)CSP结构

Yolov4网络结构中，借鉴了CSPNet的设计思路，在主干网络中设计了CSP结构。
![img](https://img-blog.csdnimg.cn/20210422081103364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

Yolov5与Yolov4不同点在于， **Yolov4中只有主干网络使用了CSP结构** ，而 **Yolov5中设计了两种CSP结构**，以Yolov5s网络为例，以CSP1_X结构应用于Backbone主干网络，另一种CSP2_X结构则应用于Neck中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422081111436.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

### 3、Neck

#### (1)FPN+PAN

Yolov5现在的Neck和Yolov4中一样，都采用FPN+PAN的结构，但在Yolov5刚出来时，只使用了FPN结构，后面才增加了PAN结构，此外网络中其他部分也进行了调整。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422081120275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

但如上面CSPNet中讲到， **Yolov5和Yolov4的不同点在于，Yolov4的Neck中，采用的都是普通的卷积操作。而Yolov5的Neck结构中，采用借鉴CSPNet设计的CSP2结构，加强网络特征融合的能力。**
![img](https://img-blog.csdnimg.cn/20210422081130240.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

### 4、prediction

#### (1)Bounding box 损失函数

**Yolov5中采用其中的GIOU_Loss做Bounding box的损失函数。** 而Yolov4中采用CIOU_Loss作为目标Bounding box的损失函数。

#### (2)NMS 非极大值抑制

在目标检测的后处理过程中，针对很多目标框的筛选，通常需要nms操作。Yolov4在DIOU_Loss的基础上采用DIOU_nms的方式，而**Yolov5中仍然采用加权nms的方式。** **DIOU_nms对于一些遮挡重叠的目标，确实会有一些改进。** ，后期可以优化。

### 5、 Yolov5四种网络结构的不同点

Yolov5代码中的四种网络，和之前的Yolov3，Yolov4中的cfg文件不同，都是以yaml的形式来呈现。
而且四个文件的内容基本上都是一样的，只有最上方的**depth_multiple**和**width_multiple**两个参数不同。

#### (1)四种结构的参数

![img](https://img-blog.csdnimg.cn/20210422081139140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

四种结构就是通过上面的两个参数，来进行控制网络的深度和宽度。其中 **depth_multiple控制网络的深度，width_multiple控制网络的宽度。**

#### (2)Yolov5网络结构

四种结构的yaml文件中，下方的网络架构代码都是一样的。如何控制网络的宽度和深度，yaml文件中的Head部分也是同样的原理。
![img](https://img-blog.csdnimg.cn/20210422081147672.png#pic_center)

在对网络结构进行解析时，yolo.py中下方的这一行代码将四种结构的depth_multiple width_multiple提取出，赋值给gd，gw。后面主要对这gd，gw这两个参数进行讲解。

```python
anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
1
```

下面再细致的剖析下，看是**如何控制每种结构，深度和宽度**的。

##### (1)控制深度

![img](https://img-blog.csdnimg.cn/20210422081155671.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

(1). 以Yolov5s为例，第一个CSP1中，使用了1个 **残差组件**，因此是CSP1_1。而在Yolov5m中，则增加了网络的深度，在第一个CSP1中，使用了2个残差组件，因此是CSP1_2。而Yolov5l中，同样的位置，则使用了3个残差组件，Yolov5x中，使用了4个残差组件。其余的第二个CSP1和第三个CSP1也是同样的原理。

(2). 在第二种CSP2结构中也是同样的方式，以第一个CSP2结构为例。Yolov5s组件中使用了 **2\*1=2组卷积** ，因此是CSP2_1。而Yolov5m中使用了2组，Yolov5l中使用了3组，Yolov5x中使用了4组。其他的四个CSP2结构，也是同理。Yolov5中，**网络的不断加深，也在不断增加网络特征提取和特征融合的能力。**

**控制深度的代码**
控制四种网络结构的核心代码是yolo.py中下面的代码，存在两个变量，n和gd。我们再将n和gd带入计算，看每种网络的变化结果。

```python
n = max(round(n * gd), 1) if n > 1 else n #depth gain
1
```

举例子: 我们选择最小的yolov5s.yaml和中间的yolov5l.yaml两个网络结构，将 **gd(height_multiple)** 系数带入，看是否正确。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422081212190.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

(1) yolov5s.yaml
其中height_multiple=0.33，即gd=0.33，而**n则由上面红色框中的信息获得**。以上面网络框图中的第一个CSP1为例，即上面的第一个红色框。n等于第二个数值3。而gd=0.33，带入计算代码，结果n=1。因此第一个CSP1结构内只有1个残差组件，即CSP1_1。第二个CSP1结构中，n等于第二个数值9，而gd=0.33，带入计算，结果n=3，因此第二个CSP1结构中有3个残差组件，即CSP1_3。第三个CSP1结构也是同理，这里不多说。
(2）yolov5l.yaml
其中height_multiple=1，即gd=1和上面的计算方式相同，第一个CSP1结构中，n=3，带入代码中，结果n=3，因此为CSP1_3。下面第二个CSP1结构和第三个CSP1结构都是同样的原理。

##### (2). 控制宽度

**Yolov5四种网络的宽度**

![img](https://img-blog.csdnimg.cn/20210422081221781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

四种Yolov5结构在不同阶段的 卷积核的数量 都是不一样的。因此也直接影响卷积后特征图的第三维度，即厚度，也为网络的宽度。
(1）以Yolov5s结构为例，第一个Focus结构中，最后卷积操作时，卷积核的数量是32个，**因此经过Focus结构，特征图的大小变成304×304×32。** 而Yolov5m的Focus结构中的卷积操作使用了48个卷积核，因此Focus结构后的特征图变成304×304×48。Yolov5l，Yolov5x也是同样的原理。
(2) **第二个卷积操作时，Yolov5s使用了64个卷积核，因此得到的特征图是152×152×64。** 而Yolov5m使用96个卷积核，因此得到的特征图是152×152×96。Yolov5l，Yolov5x也是同理。
(3) 后面三个卷积下采样操作也是同样的原理。**四种不同结构的卷积核的数量不同，这也直接影响网络中比如CSP1结构，CSP2等结构，以及各个普通卷积**，卷积操作时的卷积核数量也同步在调整，影响整体网络的计算量。大家最好可以将结构图和前面第一部分四个网络的特征图链接，对应查看，思路会更加清晰。当然卷积核的数量越多，特征图的厚度，即宽度越宽，网络提取特征的学习能力也越强。

**控制宽度的代码**
在Yolov5的代码中，控制宽度的核心代码是yolo.py文件里面的这一行：

```python
c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
1
```

它所调用的子函数make_divisible的功能是：

```python
def make_divisible(x, divisor):
    # Return x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor
123
```

举例子: 我们选择最小的yolov5s.yaml和中间的yolov5l.yaml两个网络结构，将 **gw(width_multiple)** 系数带入，看是否正确。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210422081230180.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1Y2hhb2h1bzcyNA==,size_16,color_FFFFFF,t_70#pic_center)

(1) yolov5s.yaml 其中width_multiple=0.5，即gw=0.5。以第一个卷积下采样为例，即Focus结构中下面的卷积操作。**按照上面Backbone的信息，我们知道Focus中，标准的c2=64，而gw=0.5，** 代入计算公式，最后的结果=32。即Yolov5s的Focus结构中，卷积下采样操作的卷积核数量为32个。 再计算后面的第二个卷积下采样操作，标准c2的值=128，gw=0.5，代入公式，最后的结果=64，也是正确的。
(2) yolov5l.yaml 其中width_multiple=1，即gw=1，而标准的c2=64，代入上面（2）的计算公式中，可以得到Yolov5l的Focus结构中，卷积下采样操作的卷积核的数量为64个，而第二个卷积下采样的卷积核数量是128个。
另外的三个卷积下采样操作，以及Yolov5m，Yolov5x结构也是同样的计算方式。
