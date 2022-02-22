# **人体姿态估计:基于飞桨复现Hourglass**

# 1.简介
本项目基于PaddlePaddle复现《Stacked Hourglass Networks for Human Pose Estimation》论文，该论文提出了一种人体姿态估计的方法，在MPII数据集上达到如下精度：

| size | mean@0.1 |
| -- | -- |
| 384x384 | 0.366 |
| 256x256 | 0.317 |

本文介绍了一种新的用于人体姿态估计的卷积网络结构。所有尺度上进行特征的处理和融合，做优地捕捉与身体相关的各种空间关系。

人体姿态估计可以应用在很多领域：

1.动作识别，可以检测一个人是否摔倒或疾病，也可以用于健身、体育舞蹈等教学任务。

2.运动捕捉，可以通过人体姿态的估计，在计算机上渲染图形，例如电影特效。

3.训练机器人，可以让机器人跟随一个做特定动作的人体骨架。

# 2.模型介绍

Hourglass网络采用沙漏形状的设计是为了在每个尺度上捕捉信息。而本地证据对于识别人脸和手等特征至关重要。最终的姿势估计需要对整个身体有一个连贯的理解。人的方位、四肢的排列以及相邻关节的关系都是在图像中不同镜头下最容易识别的众多线索之一。Hourglass是一个简单的，最小的设计，有能力捕捉所有这些功能，并将它们结合起来输出像素级的预测。网络必须有某种机制来有效地处理和巩固跨尺度的数据特征。Hourglass网络选择使用带有跳过层的单一管道来保留每个分辨率下的空间信息。该网络的最低分辨率为4x4像素，允许应用更小的空间过滤器来比较整个图像空间的特征。Hourglass的设置如下:卷积和最大池化层用于处理低分辨率的特征，在每一个最大池化，网络分支，应用更多的卷积在已经做过池化操作的分辨率上。在达到最低分辨率后，网络开始自顶向下的上采样序列和跨尺度的特征组合。为了将两个相邻分辨率的信息聚合在一起，我们遵循Tompson等人所描述的过程，对较低分辨率进行最近邻上采样，然后对两组特征进行元素相加。Hourglass的拓扑结构是对称的，所以每向下呈现一层，就有相应向上的一层。在达到网络的输出分辨率后，使用两轮连续的1x1卷积来产生最终的网络预测结果。网络的输出是一组heatmap，对于给定的heatmap，网络预测各个关节在每个像素上存在的概率。

_ _ _

#### 整个Hourglass 网络有多个Hourglass模块组成，允许重复的自底向上，自顶向下的推理预测
![](https://ai-studio-static-online.cdn.bcebos.com/d7d75c7014bf41eaa985649e296e8abe3885e1b6f159452984e93b0c23a236e0)

#### 下图是一个Hourglass的单个模块,在整个Hourglass网络中，特征的数量是一致的
![](https://ai-studio-static-online.cdn.bcebos.com/01b8f741048f40b9a3cbfc003589b9bab450a43a74ce4d71a5ee79201fa4dbf3)


#### 上图中的每一个方块都对应了一个redisdual模块，如下图所示
![](https://ai-studio-static-online.cdn.bcebos.com/c27fd9aebd234e8f82431e74d59c38fd262c4ea79e5948ceaa7c0b86d413b89f)


# 3.数据集下载

MPII:[https://aistudio.baidu.com/aistudio/datasetdetail/107551](https://aistudio.baidu.com/aistudio/datasetdetail/107551)

数据集解压。


```python
%cd /home/aistudio/data/
!tar xvf data107551/mpii.tar.gz
```

# 4.环境

PaddlePaddle == 2.1.2

python == 3.7


# 5. 训练

训练图像尺寸为256的模型。


```python
%cd /home/aistudio/paddle_pose/
!python -u train.py --dataset_root /home/aistudio/data/mpii/ --image_size 256
```

    /home/aistudio/paddle_pose
    => num_images: 14679
    => load 22246 samples
    => num_images: 2729
    => load 2958 samples
    W0927 11:25:46.927978   383 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0927 11:25:46.931859   383 device_context.cc:422] device: 0, cuDNN Version: 7.6.


训练图像尺寸为384的模型。


```python
%cd /home/aistudio/paddle_pose/
!python -u train.py --dataset_root /home/aistudio/data/mpii/ --image_size 384
```

--image_size 指定训练出入的图片分辨率，根据验收指标这里可以输入256或384。

--dataset_root 为数据集根目录，可以根据实际情况修改。

# 6.验证模型

1.预训练模型下载地址:

链接: [https://pan.baidu.com/s/13urfrTeJueuXhn4MHcrQcw](https://pan.baidu.com/s/13urfrTeJueuXhn4MHcrQcw)

提取码: w82w

2.下载模型后使用，下列命令验证模型。

验证图片为尺寸为256x256的模型：


```python
!python val.py --image_size 256  --pretrained_model ./output/256_best_model/model.pdparams --dataset_root /home/aistudio/data/mpii/
```

验证结果：
```
[EVAL] Ankle=79.87761299600484 Elbow=89.09163062349077 Head=96.65757162346522 Hip=88.41959160211289 Knee=83.8608487080676 Mean=88.71714806141036 Mean@0.1=32.10772823107419 Shoulder=95.36345108695652 Wrist=83.77702302257738
```
验证图片为尺寸为384x384的模型：


```python
!python val.py --image_size 384  --pretrained_model ./output/384_best_model/model.pdparams --dataset_root /home/aistudio/data/mpii/
```

验证结果：
```
[EVAL] Ankle=80.86913738917394 Elbow=89.89274782636988 Head=96.8281036834925 Hip=87.81370184355791 Knee=84.62623196807967 Mean=89.13869372885766 Mean@0.1=37.58782180867529 Shoulder=95.44836956521739 Wrist=84.889784060021
```

--image_size 指定训练出入的图片分辨率，根据验收指标这里可以输入256或384。

--pretrained_model  指定训练好的模型地址，可以根据实际情况修改。

--dataset_root 为数据集根目录，可以根据实际情况修改。



![](https://ai-studio-static-online.cdn.bcebos.com/647e8a48b3094803b48b12ac337957b15068d2f0ce0d42e8a33236a63feab153)


# 7.总结
以下表格是本次论文复现的结果。
| Arch  | Input Size | Mean@0.1 | pytorch Mean@0.1 |
| :--- | :--------: | :------: | :------: |
| pose_hourglass_52 | 256x256 | 0.321 | 0.317
| pose_hourglass_52 | 384x384 | 0.376 | 0.366

本次论文复现是我第一次接触人体姿态估计这个领域，通过复现过程学习了很多知识，感谢百度飞桨提供这次比赛，让我学习到了很多。
