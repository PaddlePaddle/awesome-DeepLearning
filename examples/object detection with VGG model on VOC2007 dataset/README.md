# tf-faster-rcnn



百度项目链接：[object detection on faster rcnn - 飞桨AI Studio - 人工智能学习实训社区 (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/2246026)

注：此代码非本人编写，代码作者详见./LICENSE，此处只记录我对此份代码的理解与运行的过程，代码来源链接见参考论文。



[TOC]



## 一、模型简介

样本较多，目标尺度较大时，优先考虑VGG-net。

![image-20210803211337438](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803211337438.png)

上图展示了python版本中的VGG16模型中的faster_rcnn_test.pt的网络结构，可以清晰的看到该网络对于一副任意大小PxQ的图像：

①首先缩放至固定大小MxN，然后将MxN图像送入网络；

②而Conv layers中包含了13个conv层+13个relu层+4个pooling层；

③RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，然后计算出proposals；

④而Roi Pooling层则利用proposals从feature maps中提取proposal feature送入后续全连接和softmax网络作classification（即分类proposal到底是什么object）。



## 二、数据准备

### 数据集：

VOC2007。这个数据集的txt文件记录了图片的文件名，用于把数据划分为训练集、验证集等，XML文件里面记录了每张图片的标签数据（包括目标物体区域及对应标签）等。

### 数据准备：

网上下载VOC数据集，把VOC2007数据集的各个压缩包解压到/home/aistudio/Faster-RCNN/data/VOCdevkit2007中。

网上下载vgg16预训练模型，解压缩到/home/aistudio/Faster-RCNN/data/imagenet_weights中。



## 三、模型训练

运行代码：

cd /home/aistudio/Faster-RCNN/

python train.py



训练好的模型：

![image-20210803212542854](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212542854.png)

![](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212702578.png)



## 四、模型测试及模型推理

运行代码：

cd /home/aistudio/Faster-RCNN/

python demo.py



设计一个接口函数，通过这个接口函数能够方便地对任意一个样本进行实时预测：

./demo.py

![image-20210803213440205](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803213440205.png)

demo.py运行结果：

![image-20210803212808513](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212808513.png)

![image-20210803212820168](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212820168.png)![image-20210803212829083](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212829083.png)![image-20210803212843314](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212843314.png)



![image-20210803212902517](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212902517.png)![image-20210803212911130](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212911130.png)![image-20210803212922114](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212922114.png)



![image-20210803212933299](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212933299.png)![image-20210803212939091](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210803212939091.png)

对结果的评估与分析：

由于只迭代了40000次，demo.py有错误识别的结果发生，如图004545.jpg的狗识别成马，但仍然可以看到vgg16的优越：

1、通过增加深度能有效地提升性能；

2、从头到尾只有3x3卷积与2x2池化，简洁优美；

3、卷积代替全连接，可适应各种尺寸的图片。



## 五、参考论文

https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3









