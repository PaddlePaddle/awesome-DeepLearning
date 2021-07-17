# ***\*yolov5模型介绍\****

### ***\*一、\*******\*网络结构图\****

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps1.jpg) 

图（1）yolov5网络结构图

图源：https://blog.csdn.net/nan355655600/article/details/107852353

### ***\*二、\*******\*输入端\****

2.1 Mosaic数据增强

​	小目标的AP一般比中目标和大目标低很多。而VOC2028数据集中也包含大量的小目标，比较麻烦的是小目标的分布并不均匀。因此 通过随机缩放，随机裁剪  随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好。Mosaic增强训练时，每次读取四张图片，分别对四张图片进行翻转，缩放，色域变化等，对其以随机缩放、随机裁剪、随机排布的方式进行拼接。对小目标的检测效果很不错。

2.2 自适应锚框计算

在Yolo算法中，针对不同的数据集，都会有初始设定长宽的锚框。以yolov5s为例：

anchors:

 \- [10,13, 16,30, 33,23] # P3/8

 \- [30,61, 62,45, 59,119] # P4/16

 \- [116,90, 156,198, 373,326] # P5/32

在网络训练中，网络在初始锚框的基础上输出预测框，进而和真实框groundtruth进行比对，计算两者差距，再反向更新，迭代网络参数。在Yolov3、Yolov4中，训练不同的数据集时，计算初始锚框的值是通过单独的程序运行的。在Yolov5中，将此功能嵌入到代码中（采用K均值和遗传算法），每次训练时，自适应的计算不同训练集中的最佳锚框值。

2.3 自适应图片缩放letterbox

​	在常用的目标检测算法中，不同的图片长宽都不相同，因此常用的方式是将原始图片统一缩放到一个标准尺寸，再送入检测网络中。

​	但Yolov5代码中对此进行了改进，作者认为，在项目实际使用时，很多图片的长宽比不同。因此缩放填充后，两端的黑边大小都不同，而如果填充的比较多，则存在信息冗余，影响推理速度。因此在Yolov5代码中datasets.py的letterbox函数中进行了修改，对原始图像自适应的添加最少的黑边。

### ***\*三、\*******\*Backbone\****

3.1 Focus结构

​	在图片进入Backbone之前，对图片进行切片操作，如下图，得到四张图片。将W，H信息集中到通道空间，输入通道扩充4倍，即拼接起来的图片变成12个通道，最后将得到的新图片经过卷积操作，最终得到了没有信息丢失的二倍下采样特征图。

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps2.jpg) 

图（2）切片操作示意图

3.2 CSP结构

​	采用CSP模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率。Yolov5以CSP1_X结构应用于主干网格，另一种CSP2_X结构应用于Neck中。

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps3.jpg) 

图（3）CSP1_X、CSP2_X结构图

### ***\*四、\**** ***\*Neck（FPN+PAN）\****

​	Yolov5采用FPN+PAN的结构FPN是自顶向下的，将高层的特征信息通过上采样的方式进行传递融合，得到进行预测的特征图。FPN层的后面还添加了一个自底向上的特征金字塔，其中包含两个PAN结构，自底向上传达强定位特征。

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps4.jpg) 

图（4）FPN+PAN结构

### ***\*五、\*******\*输出端\****

5.1 Bounding Box损失函数

​	Yolov5中采用GIOU_loss做损失函数。以下对损失函数进行解释：

​	![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps5.png)

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps6.png)

绿色：预测目标边界框  红色:真实目标边界框

蓝色：将两个边界框住最小矩形   蓝色面积：![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps7.png)  红绿并集的面积：![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps8.jpg)

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps9.jpg) 

图（5）GIOU交并比示意图

5.2 NMS非极大值抑制

1.对需要进行抑制的同类别bounding-boxes按照置信度conf进行降序排序。

2.取出排好序的第一项， 计算其与剩余项间交并比，获得一个交并比list。

3.判断这些交并比值是否大于 设定的抑制阈值 ，大于阈值就抑制（删掉），小于就保留。执行完一轮后，将剩下的boxes继续执行上述步骤；4. 停止条件就是 ns_boxes (需要抑制的集合中没有任何选项) 为空。

下图为对NMS非极大值抑制的理解：

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps10.jpg) 

图（6）非极大值抑制流程

### ***\*六、\*******\*算法性能\****

​	下面是Yolov5作者的算法性能测试图，是在COCO数据集上进行的测试。因此最终的四种网络结构，性能上来说各有千秋，Yolov5s网络最小，速度最少，AP精度也最低。其他的三种网络，在此基础上，不断加深加宽网络，AP精度也不断提升，但速度的消耗也在不断增加。

![img](file:///C:\Users\lenovo\AppData\Local\Temp\ksohtml14260\wps11.jpg) 

图（7）yolov5算法性能对比

 