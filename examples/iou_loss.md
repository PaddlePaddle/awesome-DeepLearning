# Intersection over Union loss（IOU_LOSS)

## 1. IoU的定义

IoU （Intersection over Union）的全称为交并比，在目标检测领域，其计算的是 “预测的边框” 和 “真实的边框” 的交集和并集的比值。

<img src="https://img-blog.csdn.net/20180922220708895?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQwNjE2MzA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" style="zoom:100%;" />

​                                                                                                        图1. IoU的计算

## 2. 相关工作

在经典的two-stage网络**Faster-RCNN中**，由于anchor boxes的尺度和纵横比是预先设计好的，所以RPN难以处理具有较大形状变化的候选对象，尤其是对于小目标的检测效果不好。

另一经典网络**DenseBox**利用特征图的每个像素来回归一个四维距离向量（当前像素和包含它的候选物体的四个边界之间的距离），在 L2_LOSS下,四边距离被优化为四个独立的变量。但是根据直觉，这些变量应该是相互关联的。这也导致了定位不准确的问题。

此外，为了平衡不同比例的边界框，在训练是需要将图像大小调整到一个固定的比例。 因此，**DenseBox**必须使用图像金字塔，这也使得此网络速度变慢。



## 3. IoU_LOSS的提出

IoU_LOSS 直接预测边界框和真实边界框之间的最大重叠，并将所有变量作为一个整体进行联合回归，这意味着通过不断的训练迭代，神经网络能够使用任意形状和尺度来定位目标。

<img src="https://img-blog.csdnimg.cn/20191208143510666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU1ODIyNw==,size_16,color_FFFFFF,t_70" style="zoom:100%;" />

## 4. Iou的前向计算



<img src="https://img-blog.csdnimg.cn/20191208144452271.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTU1ODIyNw==,size_16,color_FFFFFF,t_70" style="zoom:100%;" />

IoU_LOSS的前向计算如上图所示，需要注意的是，在0≤IoU≤1的情况下
$$
L=-ln(IoU)
$$
实质上是IoU的交叉熵损失,因为我们可以把IoU看作一种从伯努利分布的随机变量
$$
P(IoU)=1
$$
变量IoU的交叉熵损失是
$$
L=-Pln(IoU)-(1-P)ln(1-IoU)
$$
与L2_LOSS相比，我们可以看到，IoU_LOSS并不是独立地优化四个坐标，而是将bbox视为一个整体。 因此，IoU_LOSS可以提供比L2_LOSS更加准确的边界预测。

对于输入的任意样本，IoU的值均介于$[0,1]$之间，这种自然的归一化损失使模型具有更强的处理多尺度图像的能力。

## 5. IoU的反向传播

IoU的反向传播推导如下：
$$
\nabla _x X=\frac{∂X}{∂x_t(or ∂x_{b})}=x_l+x_r
$$


$$
\nabla_x I= \frac{∂X}{∂x_l(or ∂x_{r})}=x_t+x_b
$$

$$
\frac{∂I}{∂x_t(or ∂x_{b})}=\begin{cases}
\ I_w, & if x_t<\tilde{x_t}(or x_b<\tilde{x_b})   \\
\ 0 , &otherwise \\
\end{cases}
$$


$$
\frac{∂I}{∂x_l(or ∂x_r)}=\begin{cases}
\ I_h, & if x_l<\tilde{x_l}(or x_r<\tilde{x_r})   \\
\ 0 , &otherwise \\
\end{cases}
$$

$$
\frac{∂L}{∂x}=\frac{I(\nabla_xX-\nabla_xI)-U\nabla_xI}{U^2IoU}
=\frac{1}{U}\nabla_xX-\frac{U+I}{UI}\nabla{_x}I
$$

分析上式可知，loss值与$\nabla_xX $成正比，因此当神经网络模型预测的面积越大，损失也越大。

同时loss和$\nabla_xI$成反比，因此使用IoU_LOSS会使交集尽可能的大。

综合上述两点，可知当bounding box等于ground truth时检测效果最好。

## 6. IoU_LOSS的缺点

IoU_LOSS无法避免的缺点是：当两个box无交集时，IoU=0，很近的无交集框和很远的无交集框的输出一样，这样就失去了梯度方向，无法优化。因此，后续研究人员提出了GIoU、CIoU、DIoU等优化后的损失函数来替代IoU_LOSS



## 7. Reference

https://arxiv.org/abs/1608.01471