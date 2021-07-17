### 1 YOLO系列算法

YOLO(You Only Look Once)算法是2015年由Joseph Redmon等人提出的一种新的基于深度学习的图像检测算法。Yolo算法将图像检测作为一个在空间上分离的边框（bounding boxes）和关联类概率的回归问题。一个单一神经网络可以只用一次评估（look once）就从完整图像中预测出边框和类概率。

算法的原理是，当图像被导入后，Yolo的cnn网络将图片分割成S×S网格，如果某个物体的中心落在这个网格中，则这个网格就负责预测这个物体。每个网络需要预测B个BBox的位置信息和置信度（confidence信息），每个BBox要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个BBox还要预测C个categories。输出就是S x S x (5*B+C)的一个tensor。在test的时候，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score,得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行NMS处理，就得到最终的检测结果。

概括来就是给定一个图像，首先将其划分为S×S的网格，对于每个网格，预测B个边框，得到7*7*2个目标窗口，然后根据阈值去除可能性较低的目标窗口，最后用NMS去除冗余的窗口。

  ![image-20210717105836681](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20210717105836681.png)                            

<center>图 1 YOLO算法结构图



Yolo算法在随后的版本中，在保持保持处理速度的基础上，从预测更准确，速度更快，识别对象更多这三个方面进行了改进。v2版本相较于v1版本采用了批量归一化、高分辨率图像分类器，并使用了先验框，同时使用聚类提取先验框的尺度信息。v3版本引入了多尺度预测并使用了更好的基础分类网络，同时分类损失采用二分类交叉损失熵。v4版本提出了一种高效而强大的目标检测模型，并改进了sota的算法使其更方便于GPU训练。v5版本则相比v4拥有更快的处理速度和更轻量的模型大小。

此外，Silmyolov3是2019年由北航研究者提出的一种更窄、更好、更快的针对无人机的目标检测算法。作者对YOLOv3的卷积层通道剪枝(以通道级稀疏化)，大幅削减了模型的计算量和参数量，剪枝后的模型运行速度约为原来的两倍，并基本保持了原模型的检测精度。

Yolo具有快速，pipline简单、背景误检率低、对于艺术类作品中的物体检测同样适用且对非自然图像物体的检测率远远高于DPM和RCNN系列检测方法的优点，但是由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。同时，虽然每个格子可以预测B个bounding box，但是最终每个格子最多只能预测出一个物体。

### 2 R-CNN系列算法

Faster r-cnn算法是2015年由何恺明等人提出的一种目标检测算法，该算法在fast r-cnn的基础上提出了RPN候选框生成算法，使目标检测速度大大提高。

Faster r-cnn算法的前身是r-cnn算法和Fast r-cnn算法。

r-cnn利用selective search 算法在图像中从上到下提取2000个左右的Region Proposal，然后将每个Region Proposal缩放(warp)成227*227的大小并输入到CNN，将CNN的fc7层的输出作为特征，再将每个Region Proposal提取的CNN特征输入到SVM进行分类最后对于SVM分好类的Region Proposal做边框回归，用Bounding box回归值校正原来的建议窗口，生成预测窗口坐标。

Fast r-cnn算法利用selective search 算法在图像中从上到下提取2000个左右的建议窗口(Region Proposal)，然后将整张图片输入CNN，进行特征提取，再把建议窗口映射到CNN的最后一层卷积feature map上，通过RoI pooling层使每个建议窗口生成固定尺寸的feature map，最后利用Softmax Loss(探测分类概率) 和Smooth L1 Loss(探测边框回归)对分类概率和边框回归(Bounding box regression)联合训练。

Faster R-CNN由两个模块组成，第一个模块是一个用来产生候选框的深度全卷积网络，第二个模块是使用之前提出的候选框做检测的Fast R-CNN检测器。整个系统是一个单独的、统一的目标检测网络。

算法的原理是，首先将整张输入图像输入到cnn网络进行特征提取，然后用RPN生成建议窗口，接着把建议窗口映射到cnn的最后一层卷积feature map上，通过Rol pooling层使每个Rol生成固定尺寸的feature map，最后利用Softmax Loss和Smooth L1 Loss对分类概率和边框回归联合训练。

![image-20210717110312930](C:\Users\86133\AppData\Roaming\Typora\typora-user-images\image-20210717110312930.png)

 

<center>图 2 Faster RCNN算法结构图



Fast r-cnn算法相比于r-cnn算法，在最后一层卷积层后增加了ROI pooling layer，且损失函数使用了多任务损失函数，将边框回归直接加入到cnn网络训练中，由此改进了r-cnn算法测试速度慢和训练速度慢的缺点。

Faster r-cnn算法与fast r-cnn算法相比，由于采用两阶段加RPN，实现了更高精度的检测性能，同时可以解决多尺度、小目标的问题。但是，RPN过程中使用的NMS后处理对有遮挡对象不友好，且算法的的原始RoI pooling两次取整会带来精度的丢失。

 