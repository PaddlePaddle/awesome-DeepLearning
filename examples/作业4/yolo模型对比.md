yolov1
（1）优点：

YOLO检测速度非常快。标准版本的YOLO可以每秒处理 45 张图像；YOLO的极速版本每秒可以处理150帧图像。这就意味着 YOLO 可以以小于 25 毫秒延迟，实时地处理视频。对于欠实时系统，在准确率保证的情况下，YOLO速度快于其他方法。
YOLO 实时检测的平均精度是其他实时监测系统的两倍。
迁移能力强，能运用到其他的新的领域（比如艺术品目标检测）。
（2）局限：

YOLO对相互靠近的物体，以及很小的群体检测效果不好，这是因为一个网格只预测了2个框，并且都只属于同一类。
由于损失函数的问题，定位误差是影响检测效果的主要原因，尤其是大小物体的处理上，还有待加强。（因为对于小的bounding boxes，small error影响更大）
YOLO对不常见的角度的目标泛化性能偏弱。

yolov2
改进方法
（1）Batch Normalization
（2）引入 Anchor Box 机制
（3）Convolution With Anchor Boxes
（4）聚类方法选择Anchors
（5）Fine-Grained Features
性能表现
在VOC2007数据集上进行测试，YOLOv2在速度为67fps时，精度可以达到76.8的mAP；在速度为40fps时，精度可以达到78.6
的mAP 。可以很好的在速度和精度之间进行权衡。下图是YOLOv1在加入各种改进方法后，检测性能的改变。可见在经过多种改进方法后，YOLOv2在原基础上检测精度具有很大的提升！
![](https://ai-studio-static-online.cdn.bcebos.com/213c5d30ebd24b9794894a347b86956d20ab08c197a2413ca6dc9bd3ded033df)

yolov3
改进之处
YOLOv3最大的改进之处还在于网络结构的改进，由于上面已经讲过。因此下面主要对其它改进方面进行介绍：
（1）多尺度预测
（2）损失函数
（3）多标签分类

性能表现
如下图所示，是各种先进的目标检测算法在COCO数据集上测试结果。很明显，在满足检测精度差不都的情况下，YOLOv3具有更快的推理速度！
![](https://ai-studio-static-online.cdn.bcebos.com/2e6439e7b5dc454482a53edf5331a3d0547ca9c6496f46918acbf3d03dfc638b)

如下表所示，对不同的单阶段和两阶段网络进行了测试。通过对比发现，YOLOv3达到了与当前先进检测器的同样的水平。检测精度最高的是单阶段网络RetinaNet，但是YOLOv3的推理速度比RetinaNet快得多。
![](https://ai-studio-static-online.cdn.bcebos.com/242f4535ed3148acbdd6a628bd5b57e9fba2c04e4e3947e0a07f6f7d83e4565e)


yolov4
改进方法
除了下面已经提到的各种Tricks，为了使目标检测器更容易在单GPU上训练，作者也提出了5种改进方法：

（1）Mosaic
（2）SAT
（3）CmBN
（４）修改过的SAM
（５）修改过的PAN

性能表现
如下图所示，在COCO目标检测数据集上，对当前各种先进的目标检测器进行了测试。可以发现，YOLOv4的检测速度比EfficientDet快两倍，性能相当。同时，将YOLOv3的AP和FPS分别提高10%和12%，吊打YOLOv3!
![](https://ai-studio-static-online.cdn.bcebos.com/2633a704161b4aa4bc9c3cc4f945ad13a2550bab7a45427e94b97a339a00a8c0)

综合以上分析，总结出YOLOv4带给我们的优点有：
与其它先进的检测器相比，对于同样的精度，YOLOv4更快（FPS）；对于同样的速度，YOLOv4更准（AP）。
YOLOv4能在普通的GPU上训练和使用，比如GTX 1080Ti和GTX 2080Ti等。
论文中总结了各种Tricks（包括各种BoF和BoS），给我们启示，选择合适的Tricks来提高自己的检测器性能。




请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
