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

可变形卷积（Deformable Convolution） 可变形卷积是指卷积核在每一个元素上额外增加了一个参数方向参数，这样卷积核就能在训练过程中扩展到很大的范围。
![](https://ai-studio-static-online.cdn.bcebos.com/5cef20abcfbc484f8ec537effcc756dd8c1534beb6284d1fb5da56a8d9dd30c1)
上图中 
（a）是传统的标准卷积核，尺寸为3x3（图中绿色的点）； 
（b）就是我们今天要谈论的可变形卷积，通过在图（a）的基础上给每个卷积核的参数添加一个方向向量（图b中的浅绿色箭头），使的我们的卷积核可以变为任意形状； 
（c）和（d）是可变形卷积的特殊形式。 
我们知道卷积核的目的是为了提取输入物的特征。我们传统的卷积核通常是固定尺寸、固定大小的（例如3x3，5x5，7x7.）。这种卷积核存在的最大问题就是，对于未知的变化适应性差，泛化能力不强。
卷积单元对输入的特征图在固定的位置进行采样；池化层不断减小着特征图的尺寸；RoI池化层产生空间位置受限的RoI。网络内部缺乏能够解决这个问题的模块，这会产生显著的问题，例如，同一CNN层的激活单元的感受野尺寸都相同，这对于编码位置信息的浅层神经网络并不可取，因为不同的位置可能对应有不同尺度或者不同形变的物体，这些层需要能够自动调整尺度或者感受野的方法。再比如，目标检测虽然效果很好但是都依赖于基于特征提取的边界框，这并不是最优的方法，尤其是对于非网格状的物体而言。
解决上述问题最直观的想法就是，我们的卷积核可以根据实际情况调整本身的形状，更好的提取输入的特征。
可变形卷积结构形式 
![](https://ai-studio-static-online.cdn.bcebos.com/bc5d1c5b68864ddba183792e60925f8baa279342db214ac6bd03fc530ca35bdb)
上图是可变形卷积的学习过程，首先偏差是通过一个卷积层获得，该卷积层的卷积核与普通卷积核一样。输出的偏差尺寸和输入的特征图尺寸一致。生成通道维度是2N，分别对应原始输出特征和偏移特征。这两个卷积核通过双线性插值后向传播算法同时学习。 
事实上，可变形卷积单元中增加的偏移量是网络结构的一部分，通过另外一个平行的标准卷积单元计算得到，进而也可以通过梯度反向传播进行端到端的学习。加上该偏移量的学习之后，可变形卷积核的大小和位置可以根据当前需要识别的图像内容进行动态调整，其直观效果就是不同位置的卷积核采样点位置会根据图像内容发生自适应的变化，从而适应不同物体的形状、大小等几何形变。然而，这样的操作引入了一个问题，即需要对不连续的位置变量求导。作者在这里借鉴了之前Spatial Transformer Network和若干Optical Flow中warp操作的想法，使用了bilinear插值将任何一个位置的输出，转换成对于feature map的插值操作。同理，类似的想法可以直接用于 (ROI) Pooling中改进。 可变形卷积网络与传统网络结构上的区别如下图所示： ![](https://ai-studio-static-online.cdn.bcebos.com/f95f9c58d742459c9b961a8c41c7757cc08581782aa64956b502ccff0f8a8f7c)
可变形卷积的学习过程 
图a是标准卷积的采样过程，图b是可变形卷积的采样过程。
![](https://ai-studio-static-online.cdn.bcebos.com/537f4bba96c84a509735439aa704c933f1b9ca538e4a43d6bb62b5a26e3f363d)
我们一层层的看： 最上面的图像是在大小不同的物体上的激活单元。 中间层是为了得到顶层激活单元所进行的采样过程，左图是标准的3x3方阵采样，右图是非标准形状的采样，但是采样的点依然是3x3. 最下面一层是为了得到中间层进行的采样区域。明显发现，可变形卷积在采样时可以更贴近物体的形状和尺寸，而标准卷积无法做到这一点。
可变形卷积实现 
可变形卷积是在传统卷积的基础上，增加了调整卷积核的方向向量，使的卷积核的形态更贴近特征物。那么这个过程是如何实现的？ ①和正常的卷积神经网络一样，根据输入的图像，利用传统的卷积核提取特征图。 ②把得到的特征图作为输入，对特征图再施加一个卷积层，这么做的目的是为了得到可变形卷积的变形的偏移量。 ③偏移层是2N，是因为我们在平面上做平移，需要改变x xx值和y yy值两个方向。 ④在训练的时候，用于生成输出特征的卷积核和用于生成偏移量的卷积核是同步学习的。其中偏移量的学习是利用插值算法，通过反向传播进行学习。 
![](https://ai-studio-static-online.cdn.bcebos.com/5daba25e7cc641c09af23ce519d8525cfb3bf3aab16c4492892e9b7219bb318b)
