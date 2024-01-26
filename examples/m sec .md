什么是可变形卷积？ 可变形卷积是指卷积核在每一个元素上额外 增加了一个参数方向参数
，这样卷积核就能在训练过程中扩展到很大的范围。

卷积神经网络（CNNs）由于卷积核固定的几何结构（常见的 1x1、3x3 和 5x5
等），导致其不能够很好地建模存在几何形变（geometric
transformations）的物体。本文提出了两个可以用于提高 CNNs
建模几何形变能力的模块——deformable convolution 和 deformable RoI pooling
，两种模块都是通过在目标任务中学习偏移量（offsets）来改变空间中的采样位置。

从下面的图 1 中，可以看出标准卷积（standard
convolutions）和可变形卷积（deformable
convolutions）之间的区别。图（a）是一个常用的 3x3
的标准卷积所对应的采样点（绿色点）。图（b）是一个可变形卷积所对应的采样点（蓝色点），其中的箭头就是本文需要学习的偏移量（offsets），根据这些偏移量，就可以把标准卷积中对应的采样点（
图(a)中绿色 ）移动到可变形卷积中的不规则采样点处（ 图(b)中蓝色
）。图（c）和图（d）是图（b）的特殊情况，表明了可变形卷积囊括了长宽比、尺度变换和旋转变换

![](media/b280094342cae5b01d7a150d5b1da4ac.png)

![论文阅读-可变形卷积v2: More Deformable, Better
Results](media/c72e58e2dabf91758c01c054b58058c4.jpeg)
