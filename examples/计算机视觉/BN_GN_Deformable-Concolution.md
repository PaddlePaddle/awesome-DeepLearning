**批归一化（BN）**
	批归一化（Batch Normalization），简称BN，BN是由Google于2015年提出，这是一个深度神经网络训练的技巧，它不仅可以加快了模型的收敛速度，而且更重要的是在一定程度缓解了深层网络中“梯度弥散（特征分布较散）”的问题，从而使得训练深层网络模型更加容易和稳定。所以目前BN已经成为几乎所有卷积神经网络的标配技巧了。
**BN算法流程**
BN可以作为神经网络的一层，放在激活函数（如Relu）之前。BN的算法流程如下图：
	![](https://ai-studio-static-online.cdn.bcebos.com/028fa27c3f7a415483e7d2e475a8ab62fe49e581fc344e3f898eee4f203b8b80)
	1、求每一个小批量训练数据的均值
	2、求每一个小批量训练数据的方差
	3、使用求得的均值和方差对该批次的训练数据做归一化，获得0-1分布。其中ε是为了避免除数为0时所使用的微小正数。
	4、尺度变换和偏移：将xi乘以γ调整数值大小，再加上β增加偏移后得到yi ，这里的γ是尺度因子，β是平移因子。这一步是BN的精髓，由于归一化后的xi基本会被限制在正态分布下，使得网络的表达能力下降。为解决该问题，我们引入两个新的参数：γ,β。γ和β是在训练时网络自己学习得到的。
**BN算法的优点**
	1、减少了人为选择参数。在某些情况下可以取消 dropout 和 L2 正则项参数,或者采取更小的 L2 正则项约束参数；
	2、减少了对学习率的要求。可以使用初始很大的学习率或者选择了较小的学习率，算法也能够快速训练收敛；
	3、可以不再使用局部响应归一化。BN 本身就是归一化网络(局部响应归一化在 AlexNet 网络中存在)
	4、破坏原来的数据分布，一定程度上缓解过拟合。
	5、减少梯度消失，加快收敛速度，提高训练精度。
**BN适用场景**
	每个mini-batch比较大，数据分布比较接近。在进行训练之前，要做好充分的shuffle，否则效果会差很多。另外，由于BN需要在运行过程中统计每个mini-batch的一阶统计量和二阶统计量，因此不适用于动态的网络结构和RNN网络。
    
**群组归一化（GN）**
	Group Normalization（GN）是针对Batch Normalization（BN）在batch size较小时错误率较高而提出的改进算法，因为BN层的计算结果依赖当前batch的数据，当batch size较小时，该batch数据的均值和方差的代表性较差，因此对最后的结果影响也较大。
	随着batch size越来越小，BN层所计算的统计信息的可靠性越来越差，这样就容易导致最后错误率的上升；而在batch size较大时则没有明显的差别。虽然在分类算法中一般的GPU显存都能cover住较大的batch设置，但是在目标检测、分割以及视频相关的算法中，由于输入图像较大、维度多样以及算法本身原因等，batch size一般都设置比较小，所以GN对于这种类型算法的改进应该比较明显。
	Group Normalization（GN）的思想并不复杂，简单讲就是要使归一化操作的计算不依赖batch size的大小，深度网络中的数据维度一般是[N, C, H, W]格式，N是batch size，H/W是feature的高/宽，C是feature的channel，压缩H/W至一个维度，自然想到能否把注意力纬度从batch这一纬转移至chanel这一纬度，Batch归一化有BD、MBGD、SGD，同样G归一化有Layer Normalization、Group Normalization、Instance Normalization，这些均按chanel大小划分，与batch纬无关。
	![](https://ai-studio-static-online.cdn.bcebos.com/8451e72f61184f6b9460e8a55f3e15ff542b3974e50b44daae00e7b1e8d48198)
   上面的图展示了四种归一化方法，蓝色的块表示用这些像素计算均值和方差，然后对它们进行归一化。 
	1、BatchNorm是在batch方向做归一化，算(N, H, W)轴上的均值和方差
	2、LayerNorm是在channel方向做归一化，算(C, H, W)轴上的均值和方差
	3、InstanceNorm是在一个批次的一个channel内做归一化，算(H,W)轴上的均值和方差
	4、GroupNorm是将channel分成几个group，然后每个group内做归一化，算((C//G),H,W)轴上的均值和方差

**GN算法流程**
	![](https://ai-studio-static-online.cdn.bcebos.com/9963f442d1394a4e888a0365de8f45b212a8b0aebe14495481d7c3e842e152c9)
   ![](https://ai-studio-static-online.cdn.bcebos.com/6c763d6f2f2546828507af7d087c734e90e0a37f725f43288d65b49175deb6df)
   对于群组归一化来说， 归一化的数据需要在同一个批次，而且要在同一个群组（channel方向）
**GN作用**
	随着batch size越来越小，BN层所计算的统计信息的可靠性越来越差，这样就容易导致最后错误率的上升。但是在目标检测、分割以及视频相关的算法中，由于输入图像较大、维度多样以及算法本身原因等，batch size一般都设置比较小，所以GN对于这种类型算法的改进应该比较明显。
**GN应用场景**
	BN对批大小的选择十分敏感，批大小减小时，分类错误率显著增加；但是GN对批大小不敏感，无论批大小是多少，它的错误率都差不多。批大小越小，BN和GN之间的性能差异越明显。
   GN适用于在批大小较小时代替BN算法进行训练。

**可变形卷积**
	可变形卷积是指卷积核在每一个元素上额外增加了一个参数方向参数，这样卷积核就能在训练过程中扩展到很大的范围。下图中图（a）是标准卷积核，其他均为可变形卷积，通过在图（a）的基础上给每个卷积核的参数添加一个方向向量（图b中的浅绿色箭头），使卷积核可以变为任意形状；
	![](https://ai-studio-static-online.cdn.bcebos.com/320e3f422ab646b9a98eab82713ff32af127d968766847f7a2cdfd458b3a9ca5)
   卷积单元对输入的特征图在固定的位置进行采样；池化层不断减小着特征图的尺寸；RoI池化层产生空间位置受限的RoI。网络内部缺乏能够解决这个问题的模块，这会产生显著的问题，例如，同一CNN层的激活单元的感受野尺寸都相同，这对于编码位置信息的浅层神经网络并不可取，因为不同的位置可能对应有不同尺度或者不同形变的物体，这些层需要能够自动调整尺度或者感受野的方法。再比如，目标检测虽然效果很好但是都依赖于基于特征提取的边界框，这并不是最优的方法，尤其是对于非网格状的物体而言。因此，我们需要卷积核可以根据实际情况调整本身的形状，更好的提取输入的特征。这就是可变形卷积的来源。
**可变形卷积的学习过程**
	下图是可变形卷积的学习过程，首先偏差是通过一个卷积层获得，该卷积层的卷积核与普通卷积核一样。输出的偏差尺寸和输入的特征图尺寸一致。生成通道维度是2N，分别对应原始输出特征和偏移特征。这两个卷积核通过双线性插值后向传播算法同时学习。
   ![](https://ai-studio-static-online.cdn.bcebos.com/1b59c67efe4e42f48550a14c9b4bcce34b943f5d14e44f06b94cbfd385f42aa3)
   以及标准卷积核可变形卷积的学习过程的对比：
   ![](https://ai-studio-static-online.cdn.bcebos.com/179fd79f80b04a2cad557ead45d7ca0bc3a95c00c2b141c38c9dd493dec19588)
   
**可变形卷积v1**
	可变形卷积，就是在传统的卷积操作上加入了一个偏移量，正是这个偏移量才让卷积变形为不规则的卷积，这里要注意这个偏移量可以是小数。
   ![](https://ai-studio-static-online.cdn.bcebos.com/8f753fd8e6d5406f8d9336735b830c29ca6cc2d7e63d4293a71217fb0114d892)偏移量的计算。
**可变形池化**
	理解了可变形卷积之后，Deformable RoIPooling（可变形池化）就比较好理解了。原始的RoIPooling在操作过程中是将RoI划分为k×k个子区域。而可变形池化的偏移量其实就是子区域的偏移。同理每一个子区域都有一个偏移，偏移量对应子区域有k×k个。与可变形卷积不同的是，可变形池化的偏移量是通过全连接层得到的。
	![](https://ai-studio-static-online.cdn.bcebos.com/f0616d029a364da2ba68d59db95c95249d3384742cdf4d79bc9cfa97fdc7b1f3)
**可变形卷积v2**
	DCN v1在实用后存在一些问题：可变形卷积有可能引入了无用的（区域）来干扰特征提取，这显然会降低算法的表现。作者也做了一个实验进行对比说明：
   ![](https://ai-studio-static-online.cdn.bcebos.com/c5a40334d5b64ff98d42fb0b8ad664971cbe15ed11254a80ba56d5b388cd66b1)
   虽然DCN v1更能覆盖整个物体，但是同时也会引入一些无关的背景，这造成了干扰，所以作者提出了三个解决方法：
	（1）More Deformable Conv Layers（使用更多的可变形卷积）。
	（2）Modulated Deformable Modules（在DCNv1基础（添加offset）上添加每个采样点的权重）
	（3）R-CNN Feature Mimicking（模拟R-CNN的feature）。
	作者发现把R-CNN和Faster RCNN的classification score结合起来可以提升performance，说明R-CNN学到的focus在物体上的feature可以解决无关上下文的问题。但是增加额外的R-CNN会使inference速度变慢很多。DCNV2里的解决方法是把R-CNN当做teacher network，让DCN V2的ROIPooling之后的feature去模拟R-CNN的feature，类似知识蒸馏的做法，下面会具体展开：
	![](https://ai-studio-static-online.cdn.bcebos.com/cc859f04ef9449b8a4ab7d907fed28d7dc95dffb5fd84a8090a0a90189912984)
   左边的网络为主网络（Faster RCNN），右边的网络为子网络（RCNN）。实现上大致是用主网络训练过程中得到的RoI去裁剪原图，然后将裁剪到的图resize到224×224大小作为子网络的输入，这部分最后提取的特征和主网络输出的1024维特征作为feature mimicking loss的输入，用来约束这2个特征的差异（通过一个余弦相似度计算，如下图所示），同时子网络通过一个分类损失进行监督学习，因为并不需要回归坐标，所以没有回归损失。在inference阶段仅有主网络部分，因此这个操作不会在inference阶段增加计算成本。


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
