# 前言
本项目为百度论文复现赛第四期《YOLOv4: Optimal Speed and Accuracy of Object Detection》论文复现冠军代码以及模型。 依赖环境：
- paddlepaddle-gpu2.1.2
- python3.7

代码在COCO2017数据集上训练，复现时在testdev上精度达到AP=41.2%，与论文一致。

# 介绍
YOLOv4是YOLO系列的第四代模型，其在保留YOLOv3检测头编解码方式的同时，通过使用更强的主干网络、更强的特征融合模块、以及更多的数据增强方式，让模型性能相比于YOLOv3有显著的提高，且推理速度仍然很快。

**论文:**
- [1] [1] Bochkovskiy A, Wang C Y, Liao H. YOLOv4: Optimal Speed and Accuracy of Object Detection[J].2020.

**参考项目：**
- [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

**项目aistudio地址：**
- notebook任务：[https://aistudio.baidu.com/aistudio/projectdetail/2479219](https://aistudio.baidu.com/aistudio/projectdetail/2479219)  即为本项目，环境均已经配置好，直接点击运行即可。

# 论文解析

### YOLOv4的整体网络图如下(在此感谢江大白的供图)
![](https://ai-studio-static-online.cdn.bcebos.com/682c033367914c8f9f48a12c649e049e9d107f25c1f746dba1586a4c7608ca39)

    YOLOv4属于各种先进算法集成的创新，比如不同领域发表的最新论文的tricks，集成到自己的算法中，却发现有出乎意料的改进。正是因为这个原因，YOLOv4的技术细节非常多，而且其本身也有多种变体(从AB大神的github库中就能看出)。本项目由于篇幅原因，会重点介绍YOLOv4中的一些最典型的创新点，完整的实现欢迎阅读我的代码。

#### 1. CSPDarkNet主干网络

    主干网络中有这样三个基本组件：
       1. CBM：Yolov4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成；
       2. Res unit：借鉴Resnet网络中的残差结构，让网络可以构建的更深；
       3. CSPX：借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concate组成。

    CSP模块先将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并，在减少了计算量的同时可以保证准确率。因此Yolov4在主干网络Backbone采用CSPDarknet53网络结构，主要有三个方面的优点：
        优点一：增强CNN的学习能力，使得在轻量化的同时保持准确性。
        优点二：降低计算瓶颈
        优点三：降低内存成本

   Yolov4的Backbone中都使用了Mish激活函数，而后面的网络则还是使用leaky_relu函数。

#### 2. 颈部网络FPN+PAN

    SPP模块能够显著增加感受野的大小，能够在不增加参数量、增加计算量很小的情况下涨点。

    FPN+PAN是一种被称为双塔的结构。FPN是通过上采样，自顶向下将高层信息与底层融合；PAN是通过下采样，自底向上将底层将底层信息与高层融合。这样能够显著提高特征的提取能力。
![](https://ai-studio-static-online.cdn.bcebos.com/14d539cdf22b4d97adcf29a1373009565c5891c8f20e43f19cfb241a6c4c80de)

#### 3. 检测头部分

    检测头在预测框的计算公式上与yolov3一致，但是loss中与location相关的部分，在YOLOv4中全部被替换为CIOU_loss。这使得预测框回归的速度和精度更高一些。

#### 4. 马赛克增强

![](https://ai-studio-static-online.cdn.bcebos.com/488049cd5f4d4407990e0ff7ee1e9cb87e74895aaf37476595729e27b1df3a9d)

    Mosaic数据增强采用了4张图片，随机缩放、随机裁剪、随机排布的方式进行拼接。
    主要有几个优点：
        1. 丰富数据集：随机使用4张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好。
        2. 减少GPU：可能会有人说，随机缩放，普通的数据增强也可以做，但作者考虑到很多人可能只有一个GPU，因此Mosaic增强训练时，可以直接计算4张图片的数据，使得Mini-batch大小并不需要很大，一个GPU就可以达到比较好的效果。

# 实现思路

本项目将YOLOv4模型拆解为总体模块、主干网络模块、颈部网络模块、检测头模块、损失函数模块，并通过config文件实现模块化的配置。具体实现可以参见我的代码。

- config文件目录 -> Paddle-YOLOv4/configs/yolov4
- 总体模块目录 -> Paddle-YOLOv4/model/modeling/architectures/
- 主干网络模块目录 -> Paddle-YOLOv4/model/modeling/backbones/
- 颈部网络模块目录 -> Paddle-YOLOv4/model/modeling/necks/
- 检测头模块目录 -> Paddle-YOLOv4/model/modeling/heads/
- 损失函数模块目录 -> Paddle-YOLOv4/model/modeling/losses/

尽管YOLOv4网络比较复杂，在网络结构的代码实现上主要就是调用paddle的API，因此可读性较好。注意损失函数的实现是与YOLOv3不同的！！(很多人在YOLOv4的loss中保留了v3中的loss_xy和loss_wh，这样是不对的)

# 运行
## 解压COCO数据集到Paddle-YOLOv4/data/目录下


```python
%cd /home/aistudio/Paddle-YOLOv4/data/
!unzip -oq /home/aistudio/data/data7122/val2017.zip
!unzip -oq /home/aistudio/data/data7122/test2017.zip
!unzip -oq /home/aistudio/data/data7122/annotations_trainval2017.zip
!unzip -oq /home/aistudio/data/data7122/image_info_test2017.zip
%cd /home/aistudio/Paddle-YOLOv4/output/
!cp /home/aistudio/data/data107066/best_model.* ./
```

    /home/aistudio/Paddle-YOLOv4/data


## 安装依赖库


```python
%cd /home/aistudio/Paddle-YOLOv4/
!pip install -r requirements.txt
```

## 评估

我已经预先将训练好的best_model.pdparams文件放在了Paddle-YOLOv4/output目录下


```python
%cd /home/aistudio/Paddle-YOLOv4/
!python eval.py -c configs/yolov4/yolov4_coco.yml
```

## 生成testdev的结果

我已经预先将训练好的best_model.pdparams文件放在了Paddle-YOLOv4/output目录下。将生成的bbox.json压缩后提交至测评服务器即可


```python
%cd /home/aistudio/Paddle-YOLOv4/
!python eval.py -c configs/yolov4/yolov4_coco_test.yml
```

## 预测


```python
%cd /home/aistudio/Paddle-YOLOv4/
!python predict.py -c configs/yolov4/yolov4_coco.yml --infer_img data/1.jpg
```

![](https://ai-studio-static-online.cdn.bcebos.com/80a5a7e14cab46f98273cb8e2b5491202b0393e1839547aba9c546171b8e8801)
