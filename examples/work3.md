# 目标检测任务中解决正负样本不均衡的手段

### 问题描述：
  以Faster RCNN为例，在RPN部分会生成20000个左右的Anchor,由于-张图中通常有10个左右的物体，导致可能只有100个左右的Anchor会是正样本，正负样本比例约为1 : 200,存在严重的不均衡。对于目标检测算法，主要需要关注的是对应着真实物体的正样本，在训练时会根据其loss来调整网络参数。相比之下，负样本对应着图像的背景，如果有大量的负样本参与训练，则会淹没正样本的损失,从而降低网络收敛的效率与检测精度。

### 正样本：标签区域内的图像区域，即目标图像块。
### 负样本：标签区域以外的图像区域，即图像背景区域。

## 1.ATSS:
通过实验分析对比来说明正负样本分配的重要性，设计了一种mean IoU的方法。该方法指出基于锚点的检测器和不带锚点的检测器之间的本质区别实际上是如何定义正训练样本和负训练样本，提出自适应训练样本选择，以根据对象的统计特征自动选择正训练样本和负训练样本。
特性：    
（1）保证了所有的正样本anchor都是在groundtruth的周围。
（2）最主要是根据不同层的特性对不同层的正样本的阈值进行了微调。
 
 ![](https://ai-studio-static-online.cdn.bcebos.com/9e558aff66ca4ff99e5b4987253ab4bf7a065248e635499ba61f923529e3c87d)
 
 
## 2.SAPD:
设计了一种利用注意力来软加权的训练策略，减少了对背景信息的锚点关注。
为了解决注意力偏差问题，我们提出了一个soft-weighting方案，基本思想是为每一个anchor point Pij 赋以权重 Wij 。对于每一个positive anchor point，该权重大小取决于其图像空间位置到对应实例中心的距离，距离越大，权重就越低。因此，远离目标（实例）中心的anchor point被抑制，注意力会更集中于接近目标中心的anchor point。对于negative anchor points，所有的权重都被设置为1，维持不变。权重公式如下：

![](https://ai-studio-static-online.cdn.bcebos.com/a75581eaf827490c9d432b5991e63821d25243c7f71c4578b29b957c475d2718)

 
## 3.AutoAssign:
将标签对齐看作一种连续问题，没有真正意义上的正负样本之分，每个特征图上都有正样本和负样本属性。只是权重不同而已。
GT框里的每个grid cell刚开始都可以认为是正样本/负样本，但会对应 着两个权重w+ 和 w- ，（1）w+（每个grid cell的正样本权重）的产生：不在GT bbox中的w+为0，对bbox GT的中心和bbox中的前景的中心学习一个offset（对那些不能很正常的目标比较有效，如：线状目标，环形目标），然后根据分类得分和定位得分得到confidence，将confidence与刚才产生的center prior进行结合即可产生w+。（2）w-（每个grid cell的负样本的权重）的产生：首先，不在GT bbox中的w-为1，w-的值是根据该点预测的框与GT的iou决定的，IOU越小，则该点的w-越大。
    如何产生W+和W-如图所示：
    
![](https://ai-studio-static-online.cdn.bcebos.com/b6ca6f8d1fdc4a5ea0b66ca72c821865f024958b8f6843798b73d79790b328eb)

    
 
## 4.DETR:
将目标检测任务视为一个图像到集合的问题，使用Hungarian algorithm来实现预测值与真值实现最大的匹配，并且是一一对应。
DETR总体思路是把检测看成一个set prediction的问题，并且使用Transformer来预测box的set。DETR 利用标准 Transformer 架构来执行传统上特定于目标检测的操作，从而简化了检测 pipeline。

 ![](https://ai-studio-static-online.cdn.bcebos.com/be9004f27462470b99aeeee0493cee72d5276730d2ff4fe8b125ec1424d80258)
 
 ![](https://ai-studio-static-online.cdn.bcebos.com/24893261eeba453087522b5feaecae3ce7a78fc1838242919660e1ad05438514)
 
 
 

