# 前言
本项目为百度论文复现赛第四期《YOLO9000: better, faster, stronger》论文复现冠军代码以及模型。 依赖环境：
- paddlepaddle-gpu2.1.2
- python3.7

代码在PascalVOC数据集上训练，复现精度达到mAP=76.86%，高于论文中所给出的76.8%。

# 介绍
YOLOv2是YOLO系列的第二代模型，其首次让YOLO系列模型基于锚框进行检测，并提出多尺度训练等方法，为后续的YOLOv3、YOLOv4、YOLOv5、PPYOLO奠定了基础。

**论文:**
- [1] Redmon J, Farhadi A. YOLO9000: better, faster, stronger[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 7263-7271.

**参考项目：**
- [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2290810](https://aistudio.baidu.com/aistudio/projectdetail/2290810)  即为本项目，环境均已经配置好，直接点击运行即可。

# 论文解析

### YOLOv2所用的技巧可以用论文中的一张图来概括

![](https://ai-studio-static-online.cdn.bcebos.com/1e7ef22837e945c8974d2114e6f1126ffb5f4df819ac409eb66b95540e6bedb8)

### 以下将对这些创新点进行具体阐述：

#### 1. Batch Normalization

    CNN在训练过程中网络每层输入的分布一直在改变, 会使训练过程难度加大，但可以通过normalize每层的输入解决这个问题。YOLOv2网络在每一个卷积层后添加batch normalization，通过这一方法，mAP获得了2%的提升。batch normalization 也有助于规范化模型，可以在舍弃dropout优化后依然不会过拟合。

#### 2. High Resolution Classifier

    目前的目标检测方法中，基本上都会使用ImageNet预训练过的模型（classifier）来提取特征。而预训练时的输入图像分辨率通常为224x224，导致分辨率不够高，给检测带来困难。

    所以YOLOv2将输入图像分辨率调大至448x448，但这又导致预训练的模型需要进行调整以适应新的输入。对于YOLOv2，作者首先对分类网络（自定义的darknet）进行了fine tune，分辨率改成448x448，在ImageNet数据集上训练10轮（10 epochs），训练后的网络就可以适应高分辨率的输入。这样通过提升输入分辨率以对预训练模型进行fine tune，mAP获得了4%的提升。

#### 3. Convolutional With Anchor Boxes

    YOLOv1是一种anchor-free的目标检测算法，而YOLO系列从YOLOv2开始就转为anchor-based算法。

    YOLOv1利用全连接层的数据完成边框的预测，导致丢失较多的空间信息，定位不准。没有anchor boxes，模型recall为81%，mAP为69.5%；加入anchor boxes，模型recall为88%，mAP为69.2%。这样看来，准确率只有小幅度的下降，而召回率则提升了7%。

![](https://ai-studio-static-online.cdn.bcebos.com/434ff840f2194c8d80546f920e78ebea553017c1bc07496d96845caf685b9857)

#### 4. New Network

    YOLOv2使用了一个新的分类网络DarkNet19作为特征提取部分，参考了前人的先进经验，比如类似于VGG，作者使用了较多的3 * 3卷积核，在每一次池化操作后把通道数翻倍。借鉴了network in network的思想，网络使用了全局平均池化（global average pooling），把1 * 1的卷积核置于3 * 3的卷积核之间，用来压缩特征。也用了batch normalization（前面介绍过）稳定模型训练。

#### 5. Dimension Priors

    在训练过程中网络也会学习调整boxes的宽高维度，最终得到准确的bounding boxes。但是，如果一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就更容易学到准确的预测位。和以前的精选boxes维度不同，作者使用了K-means聚类方法类训练bounding boxes，可以自动找到更好的boxes宽高维度。

#### 6. Location Prediction

![](https://ai-studio-static-online.cdn.bcebos.com/d5cfa11ba9114d39ab6b339bed4fd5bd1de6f7531b1b44d787152f6b5e16cb81)

    YOLOv2的预测框中心位置、宽高计算公式如上。之后的YOLOv3、YOLOv4都是沿用这一公式。

#### 7. Passthrough

    YOLO最终在13 * 13的特征图上进行预测，虽然这足以胜任大尺度物体的检测，但是用上细粒度特征的话，这可能对小尺度的物体检测有帮助。Faser R-CNN和SSD都在不同层次的特征图上产生区域建议（SSD直接就可看得出来这一点），获得了多尺度的适应性。这里使用了一种不同的方法，简单添加了一个转移层（ passthrough layer），这一层要把浅层特征图（分辨率为26 * 26，是底层分辨率4倍）连接到深层特征图。这里连接采用的方式为concat。

#### 8. Multiscale Training

    原来的YOLO网络使用固定的448 * 448的图片作为输入，现在加入anchor boxes后，输入变成了416 * 416。目前的网络只用到了卷积层和池化层，那么就可以进行动态调整（意思是可检测任意大小图片）。作者希望YOLOv2具有不同尺寸图片的鲁棒性，因此在训练的时候也考虑了这一点。

    不同于固定输入网络的图片尺寸的方法，作者在几次迭代后就会微调网络。没经过10次训练（10 epoch），就会随机选择新的图片尺寸。YOLO网络使用的降采样参数为32，那么就使用32的倍数进行尺度池化{320,352，…，608}。最终最小的尺寸为320 * 320，最大的尺寸为608 * 608。接着按照输入尺寸调整网络进行训练。

    Multiscale Training机制使得网络可以更好地预测不同尺寸的图片，意味着同一个网络可以进行不同分辨率的检测任务，在小尺寸图片上YOLOv2运行更快，在速度和精度上达到了平衡。同时其也可以防止网络过拟合。

#### 9. High Resolution Detector

    预测时，当输入图像分辨率为416x416时，mAP为76.8%；当输入图像分辨率为608x608时，mAP为78.6%。

### 最后，附上一张YOLOv2网络的整体结构图方便大家理解。

![](https://ai-studio-static-online.cdn.bcebos.com/f07f8a2b4e964be8a03dccafa2a0cb3eecf29e96e5484fbbbc14c5e267c9aed9)


# 实现思路

本项目将YOLOv2模型拆解为总体模块、主干网络模块、颈部网络模块、检测头模块、损失函数模块，并通过config文件实现模块化的配置。具体实现可以参见我的代码。

- config文件目录 -> YOLOv2/configs/yolov2
- 总体模块目录 -> YOLOv2/model/modeling/architectures/
- 主干网络模块目录 -> YOLOv2/model/modeling/backbones/
- 颈部网络模块目录 -> YOLOv2/model/modeling/necks/
- 检测头模块目录 -> YOLOv2/model/modeling/heads/
- 损失函数模块目录 -> YOLOv2/model/modeling/losses/

由于YOLOv2网络比较简单，所以网络结构的代码实现主要就是调用paddle的API，注意损失函数的实现是与YOLOv3不同的！！(很多人误以为YOLOv2和YOLOv3损失函数一样，就直接调用了YOLOv3的损失函数API，这样是不对的)

# 对比

本项目基于paddlepaddle_v2.1深度学习框架复现，对比于作者论文的代码：

- 本项目提供更加细节的训练流程，以及更加全面的预训练模型。
- 本项目提供基于aistudio平台可在线运行的项目地址，您不需要在您的机器上配置paddle环境可以直接浏览器在线运行全套流程。
- 本项目的提供模型在精度上较论文提高了0.06%，您如果将训练轮数再增大，精度还能提高得更多一些。
- 本项目代码的可读性远强于论文作者提供代码。

# 运行

## 解压VOC数据集到YOLOv2/data/目录下


```python
%cd /home/aistudio/YOLOv2/data/
!unzip -oq /home/aistudio/data/data63105/PascalVOC07_12.zip
```

    /home/aistudio/YOLOv2/data


## 安装依赖库


```python
%cd /home/aistudio/YOLOv2/
!pip install -r requirements.txt
```

## 训练

我已经将主干网络的预训练文件pretrained.pdparams文件放在了YOLOv2/output目录下


```python
%cd /home/aistudio/YOLOv2/
!python3 train.py -c configs/yolov2/yolov2_voc.yml --eval --fp16
```

## 评估

我已经预先将训练好的best_model.pdparams文件放在了YOLOv2/output目录下


```python
%cd /home/aistudio/YOLOv2/
!python3 tools/eval.py -c configs/yolov2/yolov2_voc.yml
```

## 预测


```python
%cd /home/aistudio/YOLOv2/
!python3 predict.py -c configs/yolov2/yolov2_voc.yml --infer_img data/dog.jpg -o use_gpu=False
```

![预测结果](https://ai-studio-static-online.cdn.bcebos.com/34c47cc73b7043658f65af1a354e61122fcbd0d17d7d4f8eace75ed438d0be34)



```python
%cd /home/aistudio/YOLOv2/
!python3 predict.py -c configs/yolov2/yolov2_voc.yml --infer_img data/person.jpg -o use_gpu=False
```

![](https://ai-studio-static-online.cdn.bcebos.com/ef711c780da24ca68eb158c1fbd12230d54d1fdefa054ec5becaf3b037c13e07)
