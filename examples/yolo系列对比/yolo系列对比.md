# YOLO目标检测网络对比
## YOLO V1
实现方法

YOLO把目标检测看作一个回归问题，直接用一个网络进行分类和框回归。

具体做法是：将image划分为S * S个网格，每个网格预测B个bbox的位置（x、y、w、h）、置信度（confidence为交并比）、类别概率。输出维度为S * S * （B * 5+C），C为类别数。无论网格中包含多少个boxes，每个网格只预测一组类概率。测试时，将条件类概率和预测框的置信度乘起来，表示每个box包含某类物体的置信度，这个分数可以将box中的类别可能性和预测精确度同时表示出来。

![](https://ai-studio-static-online.cdn.bcebos.com/b895a5b37c8b4c9cb896a8be88d50eb0b85562f72a494045acb7093df994f16e)

基本网络模型为GoogLe Net，但未使用它的inception模块，而是交替使用1 * 1和3 * 3卷积层

卷积层提取特征，全连接层预测类别和框位置回归，共24个卷积层，2个全连接层

![](https://ai-studio-static-online.cdn.bcebos.com/fd95e208fbf34224b402883c025914e73465557dedae475ebffc07e5f6c620eb)

损失函数（平方和损失函数）

包括4部分：框中心位置x,y损失 + 框宽高w,h损失 + 置信度confidence损失 + 分类loss

![](https://ai-studio-static-online.cdn.bcebos.com/6fd5b3756458470ea526aab3e00c8dda9545c3d41b3346a193650d732a50dede)

优点：

速度快。看作一个回归问题，不需要复杂的pipeline。
对图像有全局理解。用整个图像的特征去预测bbox，而不是像RCNN，只能候选框的特征预测bbox。
候选框的数量少很多，仅7 * 7 * 2=49个。而RCNN的selectlive search有2000个，计算量大。

缺点：

每个网格只预测2个bbox，限制了模型预测物体的数量。
多次下采样，边界框预测所使用的特征是相对粗糙的特征。
## YOLO V2
改进一：检测更多种类的目标

	利用大型分类数据集ImageNet扩大目标检测的数据种类，可以检测9000种类别的目标（YOLO1仅20种）

改进二：批标准化BN

	让梯度变大，避免梯度消失，收敛更快，训练过程更快，不是应用在整个数据集，有噪声，提高模型泛化能力

改进三：用高分辨率图像训练分类网络

	YOLOV1分类网络输入图像大小为224 * 224，目标检测网络输入图像大小为448 * 448，因此YOLO1需要同时完成目标检测任务和适应更高分辨率图像的任务。

改进四：借鉴RPN的anchor boxes，有先验知识，预测更快

改进五：用k-mean聚类算法，得到YOLO2的先验框piror boxes

	用k-mean聚类算法，让模型自动选择更合适的先验框长、宽（YOLO1是人工指定的，带有一定的主观性），自定义聚类算法的距离矩阵：centroid是聚类时被选为聚类中心的框，box是其他框。

改进六：将预测的偏移量限制在一个网格范围内，模型更稳定

	预测的是预测框中心相对于网格单元的偏移量，使用logistic将预测值限制到0-1范围内，这样框偏移就不会超过1个网络（RPN预测anchor box和预测框bbox的偏移量，有可能偏移量很大，导致模型不稳定）

网格为每个bbox预测5个偏移量：tx,ty,tw,th,to，设网格左上角偏移图像左上角的距离是cx,cy，且piror bounding（模板框）的高、宽为ph、pw。

预测框坐标计算如图：

![](https://ai-studio-static-online.cdn.bcebos.com/72fb656e725c47e0b08fe41e80ed41fa1ef23107d092417da295cdb26488399f)



改进七：提出passthrough层，有利于小目标检测

	前一层26 * 26 * 512特征图分为4份，串联成4个13 * 13 * 2048的特征图，再与后一层的13 * 13 * 1024特征图串联，得13 * 13 * 3072特征图。

改进八：多尺度输入图像进行训练

	FCN网络，不固定输入大小，分类网络模型：Darknet-19，类似vgg，最后使用全局平均池化，每个特征图得到1个值，再用全连接会少很多参数。

![](https://ai-studio-static-online.cdn.bcebos.com/e9038f4e6a844cc9ae80f5c0467036aca957c6ba25144c1b80314fc30f9e1bcf)

Darknet19：19个卷积层 + 5个池化层，最后一个全局平均池化层输出1000类别（没有使用全连接层）

目标检测网络模型

去掉分类网络最后一个1000类输出的卷积层，再加上3个3 * 3卷积层，每个3 * 3后都有1个1 * 1卷积层，最后1个3 * 3 * 512和倒数第2个3 * 3 * 1024之间添加一个passthrough层，得到更精细的结果，最后一个1 * 1层输出结果。

混合分类和检测数据集，联合训练分类、检测网络

YOLO2提出一种联合训练机制，混合来自检测和分类数据集的图像进行训练。当网络看到标记为检测的图像时，基于完整的yolov2损失函数进行反向传播。当它看到一个分类图像时，只从特定于分类的部分反向传播损失。

## YOLO V3
1、多标签检测

	每个框中可能有多个类别物体，而softmax只能用于单分类，因此换成sigmoid，sigmoid可以做多标签分类。

2、结合不同卷积层的特征，做多尺度预测

	将当前层上采样的特征图，加上上层的特征图，得到一个组合特征图，再添加一些卷积层来处理这个组合的特征图，这样可以预测更细粒度的目标。

3、网络结构（DarkNet53 = DarkNet19 + ResNet）

	结合残差思想，提取更深层次的语义信息。仍然使用连续的3×3和1×1的卷积层。通过上采样对三个不同尺度做预测。如将8*8的特征图上采样和16*16的特征图相加再次计算，这样可以预测出更小的物体。采用了步长为2的卷积层代替pooling层，因为池化层会丢失信息。

![](https://ai-studio-static-online.cdn.bcebos.com/83e7e701f20840b79a6222ad5bbae32afd7266a7de8d4a9381641fcb2efca311)

4、预测更多目标

	用k-mean均值聚类算法为每个网格预测9个模版框，样可以提高recall（YOLO2有5个，YOLO1有2个）

5、损失函数

	使用交叉熵损失函数进行类别预测
## YOLO V4
![](https://ai-studio-static-online.cdn.bcebos.com/d34a5e32c2df499c8d00ad05796a13e417a9669a713d46ce8b997a478e36635d)

1.backbone由Darknet53变成CSP Darknet53。

	base layer后面会分出来两个分支，一个分支是一个大残差边，另外一个分支会接resnet残差结构，最后会将二者经过融合。而yolov3中只有简单的残差，并没有最后的融合操作。

2.v4中在backbone后面进行了一个spp操作。

	目的：增加感受野

3.v4中经过了spp之后会经过一个PANet网络结构

	在yolov3中，由darknet53得到的（13,13,1024）的feature map经过两次上采样，途中与（26,26,512）和（52,52,256）进行concat。
	在yolov4中，不仅有特征金字塔结构，还有下采样的操作。由spp之后先经过三次conv，然后喂进PANet，先经过两次上采样，然后再经过两次下采样，途中进行特征融合，得到预测的三个head。

4.激活函数

	yolov3使用的leaky relu激活函数yolov4使用的mish激活函数
	$mish = xtanh{ln{1 + e^x}$ 
	优点：
	1.从图中可以看出他在负值的时候并不是完全截断，而是允许比较小的负梯度流入，从而保证信息流动。
	2.并且激活函数无边界这个特点，让他避免了饱和这一问题，比如sigmoid，tanh激活函数通常存在梯度饱和问题，在两边极限情况下，梯度趋近于1，而Mish激活函数则巧妙的避开了这一点。
	3.另外Mish函数也保证了每一点的平滑，从而使得梯度下降效果比Relu要好。

5.数据处理的不同

	yolov4中数据增强采用了mosaic数据增强。
	他的步骤也是比较简单的：
	选取四张图片。然后将四张图片分别放置一张画布的四个角。
	然后进行超参数的拼接。
	每张图的gt框也随之进行处理，若一张图片中的某个gt框由于拼接过程中删除，我们还要对其进行边缘优化操作。

6.学习率的不同

	在yolov3中，学习率一般都是使用衰减学习率，就是每学习多少epoch，学习率减少固定的值。
	在yolov4中，学习率的变化使用了一种比较新的方法：学习率余弦退火衰减。

## YOLO V5
相比于yolov4，yolov5在原理性方法没有太多改进。但是在速度与模型大小上比yolo4有较大提升，可以认为是通过模型裁剪后的工程化应用（即推理速度和准确率增加、模型尺寸减小）。

1、Data Augmentation

	YOLO V4使用了多种数据增强技术的组合，对于单一图片，使用了几何畸变，光照畸图像，遮挡(Random Erase，Cutout，Hide and Seek，Grid Mask ，MixUp)技术，对于多图组合，作者混合使用了CutMix与Mosaic 技术。除此之外，作者还使用了Self-Adversarial Training (SAT)来进行数据增强。
	YOLOV5会通过数据加载器传递每一批训练数据，并同时增强训练数据。数据加载器进行三种数据增强：缩放，色彩空间调整和马赛克增强。据悉YOLO V5的作者Glen Jocher正是Mosaic Augmentation的创造者，故认为YOLO V4性能巨大提升很大程度是马赛克数据增强的功劳，也许你不服，但他在YOLO V4出来后的仅仅两个月便推出YOLO V5，不可否认的是马赛克数据增强确实能有效解决模型训练中最头疼的“小对象问题”，即小对象不如大对象那样准确地被检测到。

2、Auto Learning Bounding Box Anchors-自适应锚定框

	yolo3中的锚框是预先利用kmeans定义好的，yolo4沿用了yolo3；
	yolo5锚定框是基于训练数据自动学习的。个人认为算不上是创新点，只是手动改代码改为自动运行。

3、Backbone-跨阶段局部网络(CSP，Cross Stage Partial Networks)

	YOLO V5和V4都使用CSPDarknet作为Backbone从输入图像中提取丰富的信息特征。CSPNet解决了其他大型卷积神经网络框架Backbone中网络优化的梯度信息重复问题，具体做法是：将梯度的变化从头到尾地集成到特征图中，减少了模型的参数量和FLOPS数值，既保证了推理速度和准确率，又减小了模型尺寸。
	CSPNe思想源于Densnet，复制基础层的特征映射图，通过dense block 发送副本到下一个阶段，从而将基础层的特征映射图分离出来。这样可以有效缓解梯度消失问题(通过非常深的网络很难去反推丢失信号) ，支持特征传播，鼓励网络重用特征，从而减少网络参数数量。CSPNet思想可以和ResNet、ResNeXt和DenseNet结合，目前主要有CSPResNext50 and CSPDarknet53两种改造Backbone网络。

4、Neck-路径聚合网络(PANET)

	YOLO V5和V4都使用PANET作为Neck来聚合特征。Neck主要用于生成特征金字塔，增强模型对于不同缩放尺度对象的检测，从而能够识别不同大小和尺度的同一个物体。
	在PANET之前，一直使用FPN(特征金字塔)作为对象检测框架的特征聚合层，PANET在借鉴 Mask R-CNN 和 FPN 框架的基础上，加强了信息传播。
	PANET基于 Mask R-CNN 和 FPN 框架，同时加强了信息传播。该网络的特征提取器采用了一种新的增强自下向上路径的 FPN 结构，改善了低层特征的传播。第三条通路的每个阶段都将前一阶段的特征映射作为输入，并用3x3卷积层处理它们。输出通过横向连接被添加到自上而下通路的同一阶段特征图中，这些特征图为下一阶段提供信息。同时使用自适应特征池化(Adaptive feature pooling)恢复每个候选区域和所有特征层次之间被破坏的信息路径，聚合每个特征层次上的每个候选区域，避免被任意分配。

5、Head-YOLO 通用检测层
	
    模型Head主要用于最终检测部分,它在特征图上应用锚定框，并生成带有类概率、对象得分和包围框的最终输出向量。yolo5在通用检测层，与yolo3、yolo4相同。

6、Network Architecture
	
    yolo4和yolo5基本相同的网络架构，都使用CSPDarknet53（跨阶段局部网络）作为Backbone，并且使用了PANET（路径聚合网络）和SPP（空间金字塔池化）作为Neck，而且都使用YOLO V3的Head。YOLO V5 s，m，l，x四种模型的网络结构是一样的。原因是作者通过depth_multiple，width_multiple两个参数分别控制模型的深度以及卷积核的个数。

7、Activation Function
	
    yolo5的作者使用了 Leaky ReLU 和 Sigmoid 激活函数。yolo5中中间/隐藏层使用了 Leaky ReLU 激活函数，最后的检测层使用了 Sigmoid 形激活函数。而YOLO V4使用Mish激活函数。
    
![](https://ai-studio-static-online.cdn.bcebos.com/9ac9e2bba77c49c29b652878cbf8049f42e757558c1742e39b64f70fb34ce3a7)

 8、Optimization Function
	YOLO V5的作者提供了两个优化函数Adam和SGD（默认），并都预设了与之匹配的训练超参数。
	YOLO V4使用SGD。
	YOLO V5的作者建议是，如果需要训练较小的自定义数据集，Adam是更合适的选择，尽管Adam的学习率通常比SGD低。但是如果训练大型数据集，对于YOLOV5来说SGD效果比Adam好。

9、Cost Function
	YOLO 系列的损失计算是基于 objectness score, class probability score,和 bounding box regression score.
	YOLO V5使用 GIOU Loss作为bounding box的损失。
	YOLO V5使用二进制交叉熵和 Logits 损失函数计算类概率和目标得分的损失。同时我们也可以使用fl _ gamma参数来激活Focal loss计算损失函数。
	YOLO V4使用 CIOU Loss作为bounding box的损失，与其他提到的方法相比，CIOU带来了更快的收敛和更好的性能。
