#数据集介绍
iChallenge-PM是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。
此数据集的设计初衷是可以完成分类、分割、检测等多种任务，这里我们只关注其中的分类任务.
H:高度近视High Myopia
N:正常视力Normal
P:病理性近视Pathologic
P是病理性近似，正样本，类别为1
H和N不是病理性近似，负样本，类别为0
读入数据的时候根据文件名确定样本标签

#ResNet网络结构
2015 年，ResNet 横空出世，一举斩获 CVPR 2016 最佳论文奖，而且在 Imagenet 比赛的三个任务以及 COCO 比赛的检测和分割任务上都获得了第一名。四年过去，这一论文的被引量已超 40000 次.。

我们知道，增加网络深度后，网络可以进行更加复杂的特征提取，因此更深的模型可以取得更好的结果。但事实并非如此，人们发现随着网络深度的增加，模型精度并不总是提升，并且这个问题显然不是由过拟合（overfitting）造成的，因为网络加深后不仅测试误差变高了，它的训练误差竟然也变高了。作者提出，这可能是因为更深的网络会伴随梯度消失/爆炸问题，从而阻碍网络的收敛。作者将这种加深网络深度但网络性能却下降的现象称为退化问题（degradation problem）。

ResNet中的Bottleneck结构和11卷积

ResNet50起，就采用Bottleneck结构，主要是引入1x1卷积。我们来看一下这里的1x1卷积有什么作用：

对通道数进行升维和降维（跨通道信息整合），实现了多个特征图的线性组合，同时保持了原有的特征图大小；

相比于其他尺寸的卷积核，可以极大地降低运算复杂度；

如果使用两个3x3卷积堆叠，只有一个relu，但使用1x1卷积就会有两个relu，引入了更多的非线性映射；
Basicblock和Bottleneck结构
我们来计算一下11卷积的计算量优势：首先看上图右边的bottleneck结构，对于256维的输入特征，参数数目：1x1x256x64+3x3x64x64+1x1x64x256=69632，如果同样的输入输出维度但不使用1x1卷积，而使用两个3x3卷积的话，参数数目为(3x3x256x256)x2=1179648。简单计算下就知道了，使用了1x1卷积的bottleneck将计算量简化为原有的5.9%，收益超高。
整个ResNet不使用dropout，全部使用BN。此外，回到最初的这张细节图，我们不难发现一些规律和特点：

受VGG的启发，卷积层主要是3×3卷积；

对于相同的输出特征图大小的层，即同一stage，具有相同数量的3x3滤波器;

如果特征地图大小减半，滤波器的数量加倍以保持每层的时间复杂度；（这句是论文和现场演讲中的原话，虽然我并不理解是什么意思）

每个stage通过步长为2的卷积层执行下采样，而却这个下采样只会在每一个stage的第一个卷积完成，有且仅有一次。

网络以平均池化层和softmax的1000路全连接层结束，实际上工程上一般用自适应全局平均池化 (Adaptive Global Average Pooling)；

从图中的网络结构来看，在卷积之后全连接层之前有一个全局平均池化 (Global Average Pooling, GAP) 的结构。

总结如下：

相比传统的分类网络，这里接的是池化，而不是全连接层。池化是不需要参数的，相比于全连接层可以砍去大量的参数。对于一个7x7的特征图，直接池化和改用全连接层相比，可以节省将近50倍的参数，作用有二：一是节省计算资源，二是防止模型过拟合，提升泛化能力；

这里使用的是全局平均池化，但我觉得大家都有疑问吧，就是为什么不用最大池化呢？这里解释很多，我查阅到的一些论文的实验结果表明平均池化的效果略好于最大池化，但最大池化的效果也差不到哪里去。实际使用过程中，可以根据自身需求做一些调整，比如多分类问题更适合使用全局最大池化（道听途说，不作保证）。如果不确定话还有一个更保险的操作，就是最大池化和平均池化都做，然后把两个张量拼接，让后续的网络自己学习权重使用。



```python
import os
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
import cv2
import random
```


```python
#解压数据
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data90836/training.zip
%cd /home/aistudio/work/palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data90836/validation.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data90836/valid_gt.zip
#返回家目录，生成模型文件位于/home/aistudio/
%cd /home/aistudio/
```

    unzip:  cannot find or open /home/aistudio/data/data90836/training.zip, /home/aistudio/data/data90836/training.zip.zip or /home/aistudio/data/data90836/training.zip.ZIP.
    [Errno 13] Permission denied: '/home/aistudio/work/palm/PALM-Training400/'
    /home/aistudio
    unzip:  cannot find or open PALM-Training400.zip, PALM-Training400.zip.zip or PALM-Training400.zip.ZIP.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0001.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0002.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0003.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0004.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0005.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0006.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0007.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0008.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0009.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0010.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0011.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0012.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0013.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0014.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0015.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0016.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0017.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0018.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0019.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0020.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0021.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0022.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0023.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0024.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0025.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0026.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0027.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0028.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0029.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0030.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0031.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0032.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0033.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0034.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0035.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0036.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0037.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0038.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0039.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0040.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0041.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0042.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0043.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0044.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0045.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0046.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0047.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0048.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0049.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0050.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0051.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0052.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0053.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0054.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0055.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0056.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0057.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0058.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0059.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0060.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0061.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0062.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0063.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0064.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0065.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0066.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0067.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0068.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0069.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0070.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0071.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0072.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0073.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0074.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0075.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0076.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0077.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0078.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0079.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0080.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0081.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0082.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0083.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0084.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0085.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0086.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0087.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0088.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0089.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0090.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0091.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0092.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0093.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0094.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0095.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0096.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0097.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0098.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0099.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0100.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0101.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0102.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0103.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0104.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0105.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0106.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0107.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0108.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0109.jpg.
    checkdir error:  cannot create /home/aistudio/work/palm/PALM-Validation400
                     Permission denied
                     unable to process PALM-Validation400/V0110.jpg.
    /home/aistudio



```python
#将数据集转换为csv格式并放到home文件夹中
import pandas as pd

data_xls = pd.read_excel('/home/aistudio/work/palm/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx', index_col=0)
data_xls.to_csv('/home/aistudio/labels.csv')
```


```python
#训练集路径
DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
#加载数据集，将数据集划分为训练集和测试集
def load_data(ratio=0.8):
    filenames = os.listdir(DATADIR)
    testdata=filenames[int(len(filenames)*ratio):-1]
    traindata=filenames[:int(len(filenames)*ratio)]
    return traindata,testdata
```


```python
# 对读入的图像数据进行预处理
def transform_img(img):
    # 将图片尺寸缩放道 224x224
    img = cv2.resize(img, (224, 224))
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img / 255.
    img = img * 2.0 - 1.0
    return img
```


```python
# 定义训练集数据读取器
def data_loader(datadir,filenames, batch_size=10, mode = 'train'):
    # 将datadir目录下的文件列出来，每条文件都要读入
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise('Not excepted file name')
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

```


```python
# 定义验证集数据读取器
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47
    # 打开包含验证集标签的csvfile，并读入其中的内容
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            # 根据图片文件名加载图片，并对图像数据作预处理
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                # 当数据列表的长度等于batch_size的时候，
                # 把这些数据当作一个mini-batch，并作为数据生成器的一个输出
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            # 剩余样本数目不足一个batch_size的数据，一起打包成一个mini-batch
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader
```


```python
#定义神经网络的训练
import os
import random
import paddle
import numpy as np

DATADIR = '/home/aistudio/work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/work/palm/PALM-Validation400'
CSVFILE = '/home/aistudio/labels.csv'

# 定义训练过程
def train_pm(model, optimizer,filenames):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start training ... ')
    model.train()
    epoch_num = 5
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR,filenames, batch_size=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            # 计算sigmoid后的预测概率，进行loss计算
            pred = F.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            # 计算预测概率小于0.5的类别
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿第一个维度级联
            pred = paddle.concat([pred2, pred], axis=1)
            acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()

        paddle.save(model.state_dict(), 'palm.pdparams')
        paddle.save(optimizer.state_dict(), 'palm.pdopt')

```


```python
# 定义评估过程
def evaluation(model, filenames,params_file_path):

    # 开启0号GPU预估
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start evaluation .......')

    #加载模型参数
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()
    eval_loader = data_loader(DATADIR, filenames,
                        batch_size=10, mode='eval')

    acc_set = []
    avg_loss_set = []
    
    for batch_id, data in enumerate(eval_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
        # 计算sigmoid后的预测概率，进行loss计算
        pred = F.sigmoid(logits)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        # 计算预测概率小于0.5的类别
        pred2 = pred * (-1.0) + 1.0
        # 得到两个类别的预测概率，并沿第一个维度级联
        pred = paddle.concat([pred2, pred], axis=1)
        acc = paddle.metric.accuracy(pred, paddle.cast(label, dtype='int64'))

        acc_set.append(acc.numpy())
        avg_loss_set.append(loss.numpy())
    print("[validation] accuracy/loss: {}/{}".format(np.mean(acc_set), np.mean(avg_loss_set)))
```


```python
#加载测试数据
def load_p_data(datadir,filenames,num_begin,num_end):
        # 将datadir目录下的文件列出来，每条文件都要读入
        def reader():
            batch_imgs = []
            batch_labels = []
            name_set=[]
            for name in filenames[num_begin:num_end]:
                filepath = os.path.join(datadir, name)
                img = cv2.imread(filepath)
                img = transform_img(img)
                if name[0] == 'H' or name[0] == 'N':
                    # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                    # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                    label = 0
                elif name[0] == 'P':
                    # P开头的是病理性近视，属于正样本，标签为1
                    label = 1
                else:
                    raise('Not excepted file name')
                # 每读取一个样本的数据，就将其放入数据列表中
                batch_imgs.append(img)
                batch_labels.append(label)
                name_set.append(name)

            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array,name_set
        return reader
```


```python
#在数据集中抽取选中的数据作预测
def predict(model,datadir,filenames,num_begin,num_end,params_file_path):
    # 开启0号GPU预估
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    print('start test .......')

    #加载模型参数
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    test_loader=load_p_data(datadir,filenames,num_begin,num_end)
    for batch_id, data in enumerate(test_loader()):
        x_data, y_data, name_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        logits = model(img)
        # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
        pred = F.sigmoid(logits)
        t = np.array(pred)
        t=np.around(t)
        t=t.reshape(1,-1)
        print(name_data)
        print(t)
```


```python
#定义神经网络 ResNet50
# -*- coding:utf-8 -*-
# ResNet模型代码
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# ResNet中使用了BatchNorm层，在卷积层的后面加上BatchNorm以提升数值稳定性
# 定义卷积批归一化块
class ConvBNLayer(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
       
        """
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)
        
        self.act = act

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y, negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y

# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状与输入不一致，则对输入图片做1x1卷积，将其输出形状调整成一致
class BottleneckBlock(paddle.nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()
        # 创建第一个卷积层 1x1
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu')
        # 创建第二个卷积层 3x3
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # 创建第三个卷积 1x1，但输出通道数乘以4
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        # 如果conv2的输出跟此残差块的输入数据形状一致，则shortcut=True
        # 否则shortcut = False，添加1个1x1的卷积作用在输入数据上，使其形状变成跟conv2一致
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        # 如果shortcut=True，直接将inputs跟conv2的输出相加
        # 否则需要对inputs进行一次卷积，将形状调整成跟conv2输出一致
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y

# 定义ResNet模型
class ResNet(paddle.nn.Layer):
    def __init__(self, layers=50, class_dim=1):
        """
        
        layers, 网络层数，可以是50, 101或者152
        class_dim，分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.layers = layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            #ResNet50包含多个模块，其中第2到第5个模块分别包含3、4、6、3个残差块
            depth = [3, 4, 6, 3]
        elif layers == 101:
            #ResNet101包含多个模块，其中第2到第5个模块分别包含3、4、23、3个残差块
            depth = [3, 4, 23, 3]
        elif layers == 152:
            #ResNet152包含多个模块，其中第2到第5个模块分别包含3、8、36、3个残差块
            depth = [3, 8, 36, 3]
        
        # 残差块中使用到的卷积的输出通道数
        num_filters = [64, 128, 256, 512]

        # ResNet的第一个模块，包含1个7x7卷积，后面跟着1个最大池化层
        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=7,
            stride=2,
            act='relu')
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1)

        # ResNet的第二到第五个模块c2、c3、c4、c5
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1, # c3、c4、c5将会在第一个残差块使用stride=2；其余所有残差块stride=1
                        shortcut=shortcut))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在c5的输出特征图上使用全局池化
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        
        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积和全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                      weight_attr=paddle.ParamAttr(
                          initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, [y.shape[0], -1])
        y = self.out(y)
        return y
```


```python
#进入保存参数的目录
%cd /home/aistudio/
#训练网络
#加载数据
traindata,testdata=load_data()
model = ResNet()
# 定义优化器
opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters(), weight_decay=0.001)
# 启动训练过程
train_pm(model, opt,traindata)
#启动评估过程
evaluation(model,testdata, params_file_path="palm.pdparams")
#模型预测，取指定的测试集预测
predict(model,DATADIR,testdata,num_begin=0,num_end=-1,params_file_path="palm.pdparams")
```

    /home/aistudio
    start training ... 


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
      "When training, we now always track global mean and variance.")


    epoch: 0, batch_id: 0, loss is: [0.67699844]
    epoch: 0, batch_id: 10, loss is: [0.7729627]
    epoch: 0, batch_id: 20, loss is: [0.7365789]
    epoch: 0, batch_id: 30, loss is: [0.59878033]
    [validation] accuracy/loss: 0.7599999308586121/0.5259617567062378
    epoch: 1, batch_id: 0, loss is: [0.434448]
    epoch: 1, batch_id: 10, loss is: [0.40699005]
    epoch: 1, batch_id: 20, loss is: [0.49765036]
    epoch: 1, batch_id: 30, loss is: [0.33471498]
    [validation] accuracy/loss: 0.8100000619888306/0.4256131052970886
    epoch: 2, batch_id: 0, loss is: [0.28392982]
    epoch: 2, batch_id: 10, loss is: [0.6830715]
    epoch: 2, batch_id: 20, loss is: [0.5263477]
    epoch: 2, batch_id: 30, loss is: [0.7211212]
    [validation] accuracy/loss: 0.8999999761581421/0.27760475873947144
    epoch: 3, batch_id: 0, loss is: [0.30452743]
    epoch: 3, batch_id: 10, loss is: [0.1176555]
    epoch: 3, batch_id: 20, loss is: [1.0202836]
    epoch: 3, batch_id: 30, loss is: [0.10208807]
    [validation] accuracy/loss: 0.9024999737739563/0.2556475102901459
    epoch: 4, batch_id: 0, loss is: [0.22765188]
    epoch: 4, batch_id: 10, loss is: [0.29931164]
    epoch: 4, batch_id: 20, loss is: [0.3710094]
    epoch: 4, batch_id: 30, loss is: [0.6657139]
    [validation] accuracy/loss: 0.8975000381469727/0.2646353244781494
    start evaluation .......
    [validation] accuracy/loss: 0.7847222089767456/0.4193152189254761
    start test .......
    ['P0191.jpg', 'N0015.jpg', 'P0162.jpg', 'P0008.jpg', 'P0159.jpg', 'N0121.jpg', 'P0103.jpg', 'H0007.jpg', 'P0098.jpg', 'P0122.jpg', 'P0209.jpg', 'P0207.jpg', 'P0152.jpg', 'P0113.jpg', 'N0111.jpg', 'N0047.jpg', 'P0056.jpg', 'P0195.jpg', 'N0113.jpg', 'P0198.jpg', 'N0080.jpg', 'P0104.jpg', 'N0049.jpg', 'P0053.jpg', 'H0006.jpg', 'N0061.jpg', 'P0023.jpg', 'N0114.jpg', 'N0091.jpg', 'N0096.jpg', 'P0072.jpg', 'P0035.jpg', 'N0019.jpg', 'P0030.jpg', 'N0064.jpg', 'P0066.jpg', 'P0026.jpg', 'P0102.jpg', 'P0045.jpg', 'P0076.jpg', 'P0090.jpg', 'P0042.jpg', 'N0032.jpg', 'P0029.jpg', 'H0003.jpg', 'P0044.jpg', 'N0068.jpg', 'P0146.jpg', 'P0194.jpg', 'P0106.jpg', 'P0108.jpg', 'P0143.jpg', 'N0093.jpg', 'N0125.jpg', 'N0054.jpg', 'N0140.jpg', 'H0026.jpg', 'N0076.jpg', 'P0184.jpg', 'P0006.jpg', 'N0018.jpg', 'N0055.jpg', 'N0025.jpg', 'P0073.jpg', 'N0031.jpg', 'N0071.jpg', 'P0173.jpg', 'P0149.jpg', 'P0144.jpg', 'P0063.jpg', 'P0175.jpg', 'P0069.jpg', 'P0022.jpg', 'P0074.jpg', 'P0211.jpg', 'H0009.jpg', 'N0006.jpg', 'P0027.jpg']
    [[1. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 1. 0. 1. 0. 0. 1. 1. 0. 1. 0. 1. 0. 1.
      0. 0. 1. 0. 0. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1.
      0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 0.
      0. 1. 1. 0. 0. 1.]]



