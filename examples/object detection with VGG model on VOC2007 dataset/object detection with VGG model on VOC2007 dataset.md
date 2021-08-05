# 实训作业



[TOC]



## 一、理论讲解

经过R-CNN和Fast RCNN的积淀，Ross B. Girshick在2016年提出了新的Faster RCNN，在结构上，Faster RCNN已经将特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。



## 二、代码实现

#### **1）** **实验设计逻辑**

任务：实现目标检测：使用VOC2007数据集，基于faster-rcnn实现目标检测。

实验设计逻辑：

![image-20210805184105340](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184105340.png)

如上图，Faster RCNN其实可以分为4个主要内容：

①Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

②Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。

③Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

④Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。



#### **2）** **数据处理**

数据集：VOC2007。这个数据集的txt文件记录了图片的文件名，用于把数据划分为训练集、验证集等，XML文件里面记录了每张图片的标签数据（包括目标物体区域及对应标签）等。

数据预处理：

./lib/datasets/factory.py

get_imdb()函数在 ./lib/datasets/factory.py 文件中，按照各个数据集的year、name分为好几个数据集，这些数据集都会放在 _sets = {} 字典中。

./lib/datasets/pascal_voc.py

pascal_voc是一个类，它继承了imdb类，在之前的输入参数下使用”trainval”、”2007”进行初始化，其中

![image-20210805184153977](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184153977.png)

确定了VOC_2007数据集的路径，而以下语句写明了类别，而且给每个类别都编了一个号。

![image-20210805184208454](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184208454.png)

self._image_index读取了 ./VOCdevkit2007/VOC2007/ImageSets/Main/train_val.txt中的所有照片的名字（不包含后缀名），之后读取每张图像的ground-truth boxes信息。

./train.py

刚刚通过调用 imdb = get_imdb(args.imdb_name)，读取image database的一些基本信息，但是这些信息不足够我们训练一个faster rcnn网络，所以进一步调用了 roidb = get_training_roidb(imdb)，进一步丰富数据的信息。

然后就开始准备数据了 prepare_roidb(imdb)。在rdl_roidb.prepare_roidb(imdb) 中，进一步丰富了imdb.roidb的内容。

![image-20210805184245379](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184245379.png)

到此为止，数据的预处理就已经完成了。

 

#### **3）** **模型设计**

样本较多，目标尺度较大时，优先考虑VGG-net。

![image-20210805184308165](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184308165.png)

上图展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，可以清晰的看到该网络对于一副任意大小PxQ的图像：

①首先缩放至固定大小MxN，然后将MxN图像送入网络；

②而Conv layers中包含了13个conv层+13个relu层+4个pooling层；

③RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，然后计算出proposals；

④而Roi Pooling层则利用proposals从feature maps中提取proposal feature送入后续全连接和softmax网络作classification（即分类proposal到底是什么object）。



#### **4）** **训练配置**

./lib/config/config.py

定义模型训练的超参数：

权重衰减weight_decay=0.0005；

学习率learning_rate=0.001；

动量momentum=0.9；

Gamma指数gamma=0.1；

批尺寸batch_size=256；

最大迭代次数max_iters=40000；

步长（目前仅支持一步）step_size=30000；

在命令行界面上显示训练期间损失的迭代间隔display=10；

缩放输入图像最长边的最大像素大小max_size=1000；

每个小批量要使用的图像ims_per_batch=1；

拍摄快照的迭代次数snapshot_iterations=5000；

……

模型实例化：./lib/nets/vgg16.py

选择初始值设定项:

![image-20210805184350671](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184350671.png)

建立预测:

![image-20210805184408452](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184408452.png)

搭建vgg网络：

![image-20210805184421937](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184421937.png)

搭建rpn网络：

![image-20210805184448463](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184448463.png)

定义优化器：./train.py

![image-20210805185227025](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805185227025.png)



#### **5）** **模型训练与评估**

训练好的模型：

![image-20210805184518868](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184518868.png)

模型评估：VGG16由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling（最大化池）分开，所有隐层的激活单元都采用ReLU函数，结构简洁。VGG16使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。VGG16网络第一层的通道数为64，后面每层都进行了翻倍，最多到512个通道，通道数的增加，使得更多的信息可以被提取出来。VGG16在网络测试阶段将训练阶段的三个全连接替换为三个卷积，使得测试得到的全卷积网络因为没有全连接的限制，因而可以接收任意宽或高为的输入，这在测试阶段很重要。



#### **6）** **模型推理**

设计一个接口函数，通过这个接口函数能够方便地对任意一个样本进行实时预测：

./demo.py

![image-20210805184605436](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210805184605436.png)