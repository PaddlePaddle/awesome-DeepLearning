# 基于ResNet网络眼疾识别

## 眼疾识别数据集iChallenge-PM

关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。如今近视已经成为困扰人们健康的一项全球性负担，在近视人群中，有超过35%的人患有重度近视。近视会拉长眼睛的光轴，也可能引起视网膜或者络网膜的病变。随着近视度数的不断加深，高度近视有可能引发病理性病变，这将会导致以下几种症状：视网膜或者络网膜发生退化、视盘区域萎缩、漆裂样纹损害、Fuchs斑等。因此，及早发现近视患者眼睛的病变并采取治疗，显得非常重要。数据可以从AIStudio下载。
数据集准备
该项目包括如下三个文件，可以点击这里进行下载，解压缩后存放在/home/aistudio/work/palm目录下。

training.zip：包含训练中的图片和标签
validation.zip：包含验证集的图片
valid_gt.zip：包含验证集的标签
valid_gt.zip文件解压缩之后，需要将PM_Label_and_Fovea_Location.xlsx文件转存成csv格式，本节代码示例中已经提前转成文件labels.csv。

## ResNet网络模型

ResNet是2015年ImageNet比赛的冠军，将识别错误率降低到了3.6%，这个结果甚至超出了正常人眼识别的精度。
Kaiming He等人提出了残差网络ResNet来解决上述问题，其基本思想如 图1所示。

图1(a)：表示增加网络的时候，将x映射成y = F ( x ) 输出。
图1(b)：对图1(a)作改进，输出y = F ( x ) + x 。这时不是直接学习输出特征y的表示，而是学习y − x 。
如果想学习出原模型的表示，只需将F(x)的参数全部设置为0，则y = x 是恒等映射。
F ( x ) = y − x 也叫做残差项，如果x → y 的映射接近恒等映射，图1(b)中通过学习残差项也比图1(a)学习完整映射形式更加容易。

![](.\image\d2e891d19d39480fa9777e264a98dbb5bfe41964d004422da299f27c37c211fc.png)

​                                                           图1：残差块设计思想

图1(b)的结构是残差网络的基础，这种结构也叫做残差块（residual block）。输入x通过跨层连接，能更快的向前传播数据，或者向后传播梯度。残差块的具体设计方案如 **图**2 所示，这种设计方案也成称作瓶颈结构。

![](.\image\322b26358d43401ba81546dd134a310cfb11ecafb3314aab88b5885ff642870b.png)

​                                                                    图2：残差块结构示意图

下图表示出了ResNet-50的结构，一共包含49层卷积和1层全连接，所以被称为ResNet-50。

![](.\image\b31389ddfdc84276873c2fc3ee5ae149e96cd1f0edf84466a35661959bbcb3dd.png)

## 参考文献

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im- age recognition. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016a.
[2] 百度架构师手把手带你实践深度学习–计算机视觉（上）：神经网络基础。
