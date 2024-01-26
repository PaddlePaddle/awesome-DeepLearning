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

群组归一化（Group Normalization）
试图以群组方式实现快速训练神经网络，这种方法对于硬件的需求大大降低，并在实验中超过了传统的批量归一化方法。
GN 将通道分成组，并在每组内计算归一化的均值和方差。GN 的计算与批量大小无关，并且其准确度在各种批量大小下都很稳定。在 ImageNet 上训练的 ResNet-50 上，GN 使用批量大小为 2 时的错误率比 BN 的错误率低 10.6％;当使用典型的批量时，GN 与 BN 相当，并且优于其他标归一化变体。而且，GN 可以自然地从预训练迁移到微调。在进行 COCO 中的目标检测和分割以及 Kinetics 中的视频分类比赛中，GN 可以胜过其竞争对手，表明 GN 可以在各种任务中有效地取代强大的 BN。在最新的代码库中，GN 可以通过几行代码轻松实现。
GN 不用批量维度，其计算与批量大小无关。GN 在大范围的批量下运行都非常稳定（图 1）。
![](https://ai-studio-static-online.cdn.bcebos.com/0eb0c23f4f444332b39f786de991f14f80607a43c61442518f858454ee6f2573)
在批量大小为 2 的样本中，GN 比 ImageNet 中的 ResNet-50 的 BN 对应的误差低 10.6％。对于常规的批量规格，GN 与 BN 表现相当（差距为 0.5％），并且优于其它归一化变体 。此外，尽管批量可能会发生变化，但 GN 可以自然地从预训练迁移到微调。在 COCO 目标检测和分割任务的 Mask R-CNN 上，以及在 Kinetics 视频分类任务的 3D 卷积网络上，相比于 BN 的对应变体，GN 都能获得提升或者超越的结果。GN 在 ImageNet、COCO 和 Kinetics 上的有效性表明 GN 是 BN 的有力竞争者，而 BN 在过去一直在这些任务上作为主导方法。
GN 在检测，分割和视频分类方面的改进表明，GN 对于当前处于主导地位的 BN 技术而言是强有力的替代。论文中把 GN 作为一个有效的归一化层且不用开发批量维度，同时也评估了 GN 在各种应用中的行为表现。不过，论文作者也注意到，由于 BN 之前拥有很强的影响力，以至于许多先进的系统及其超参数已被设计出来。这对于基于 GN 的模型可能是不利的，不过也有可能重新设计系统或搜索 GN 的新超参数将会产生更好的结果。
计算机视觉任务（包括检测、分割、视频识别和其他基于此的高级系统）对批量大小的限制要求更高。例如，Fast / er 和 Mask R-CNN 框架使用批量为 1 或 2 的图像，为了更高的分辨率，其中 BN 通过变换为线性层而被「固定」；在 3D 卷积视频分类中，时空特征的出现导致在时间长度和批大小之间需要作出权衡。BN 的使用通常要求这些系统在模型设计和批大小之间作出妥协。
群组归一化（GN）作为 BN 的替代方案。作者注意到像 SIFT 和 HOG 这样的许多经典特征是分组特征并且包括分组规范化。例如，HOG 矢量是几个空间单元的结果，其中每个单元由归一化方向直方图表示。同样，作者提出 GN 作为一个层，将通道划分为组，并对每个组内的特征进行归一化。

tensorflow代码实现


```python
def GN(x, gamma, beta, G, eps = 1e-5):
    #x:输入特征
    #gamma beta：比例和偏置
    #G: GN的群组数量

    N, C, H, W = x.shape
    x= tf.reshape(x,[N,G,C // G,H,W])

    mean,var = tf.nn.moment(x,[2,3,4],keep_dims = True)
    x = (x - mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [N,C,H,W])

    return x * gamma + beta
```

可变形卷积（Deformable Convolution）
可变形卷积是指卷积核在每一个元素上额外增加了一个参数方向参数，这样卷积核就能在训练过程中扩展到很大的范围。
![](https://ai-studio-static-online.cdn.bcebos.com/0494ee81bc124ef4885971a2eb6fe0627b5beda85ba24c63a3d362195fdda120)
上图中
（a）是传统的标准卷积核，尺寸为3x3（图中绿色的点）；
（b）就是我们今天要谈论的可变形卷积，通过在图（a）的基础上给每个卷积核的参数添加一个方向向量（图b中的浅绿色箭头），使的我们的卷积核可以变为任意形状；
（c）和（d）是可变形卷积的特殊形式。
我们知道卷积核的目的是为了提取输入物的特征。我们传统的卷积核通常是固定尺寸、固定大小的（例如3x3，5x5，7x7.）。这种卷积核存在的最大问题就是，对于未知的变化适应性差，泛化能力不强。
卷积单元对输入的特征图在固定的位置进行采样；池化层不断减小着特征图的尺寸；RoI池化层产生空间位置受限的RoI。网络内部缺乏能够解决这个问题的模块，这会产生显著的问题，例如，同一CNN层的激活单元的感受野尺寸都相同，这对于编码位置信息的浅层神经网络并不可取，因为不同的位置可能对应有不同尺度或者不同形变的物体，这些层需要能够自动调整尺度或者感受野的方法。再比如，目标检测虽然效果很好但是都依赖于基于特征提取的边界框，这并不是最优的方法，尤其是对于非网格状的物体而言。
解决上述问题最直观的想法就是，我们的卷积核可以根据实际情况调整本身的形状，更好的提取输入的特征。
可变形卷积结构形式
![](https://ai-studio-static-online.cdn.bcebos.com/527f2a19f01e4032a7556cd4ada1a9ea3e92ac09ebb7411cb536bd0f10063395)
上图是可变形卷积的学习过程，首先偏差是通过一个卷积层获得，该卷积层的卷积核与普通卷积核一样。输出的偏差尺寸和输入的特征图尺寸一致。生成通道维度是2N，分别对应原始输出特征和偏移特征。这两个卷积核通过双线性插值后向传播算法同时学习。
事实上，可变形卷积单元中增加的偏移量是网络结构的一部分，通过另外一个平行的标准卷积单元计算得到，进而也可以通过梯度反向传播进行端到端的学习。加上该偏移量的学习之后，可变形卷积核的大小和位置可以根据当前需要识别的图像内容进行动态调整，其直观效果就是不同位置的卷积核采样点位置会根据图像内容发生自适应的变化，从而适应不同物体的形状、大小等几何形变。然而，这样的操作引入了一个问题，即需要对不连续的位置变量求导。作者在这里借鉴了之前Spatial Transformer Network和若干Optical Flow中warp操作的想法，使用了bilinear插值将任何一个位置的输出，转换成对于feature map的插值操作。同理，类似的想法可以直接用于 (ROI) Pooling中改进。
可变形卷积网络与传统网络结构上的区别如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/d477398cf26848ce84ce74f6d7f70f6a9191f64fe0b54f7c9a9cf6658e5915a5)
可变形卷积的学习过程
图a是标准卷积的采样过程，图b是可变形卷积的采样过程。
![](https://ai-studio-static-online.cdn.bcebos.com/dd39b1e78f114525a122ab583a50e0c540736ac757cc4ed789f5ac380874006f)
我们一层层的看：
最上面的图像是在大小不同的物体上的激活单元。
中间层是为了得到顶层激活单元所进行的采样过程，左图是标准的3x3方阵采样，右图是非标准形状的采样，但是采样的点依然是3x3.
最下面一层是为了得到中间层进行的采样区域。明显发现，可变形卷积在采样时可以更贴近物体的形状和尺寸，而标准卷积无法做到这一点。
可变形卷积实现
可变形卷积是在传统卷积的基础上，增加了调整卷积核的方向向量，使的卷积核的形态更贴近特征物。那么这个过程是如何实现的？
①和正常的卷积神经网络一样，根据输入的图像，利用传统的卷积核提取特征图。
②把得到的特征图作为输入，对特征图再施加一个卷积层，这么做的目的是为了得到可变形卷积的变形的偏移量。
③偏移层是2N，是因为我们在平面上做平移，需要改变x xx值和y yy值两个方向。
④在训练的时候，用于生成输出特征的卷积核和用于生成偏移量的卷积核是同步学习的。其中偏移量的学习是利用插值算法，通过反向传播进行学习。
![](https://ai-studio-static-online.cdn.bcebos.com/bf99132288564ced9ff268c844da2a9176d9ee8c48e64d308343becd535cb920)

