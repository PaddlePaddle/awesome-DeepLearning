# 钢筋计数

## 1.项目说明

在工地现场，验收人员需要对入场车辆上的钢筋进行现场人工点根，确认数量无误后，才能进场卸货。上述过程繁琐、速度慢，耗费大量的人力成本
针对上述问题，飞桨PaddleX通过与摄像头结合，可以实现自动钢筋计数，再结合人工修改少量误检的方式，可以智能、高效地完成此任务。

<div align="center">
    <img src="./images/1.png" width="600"/>
</div>

除钢筋计数外，本案例中介绍的方案同样适用于密集检测的移动端部署场景，如：餐饮计数，药品计数，零件计数等。

**业务难点：**

- **精度要求高** 实际业务中，钢筋数量庞大，容易导致模型误检和漏检，需要专门针对此密集目标的检测算法进行优化；
- **钢筋尺寸不一** 钢筋的直径变化范围较大且截面形状不规则、颜色不一，拍摄的角度、距离也不完全受控，导致传统算法在实际使用的过程中效果不稳定；
- **边界难以区分** 钢筋车每次会装载很多捆钢筋，如果直接全部处理会存在边缘角度差、遮挡等问题。而使用“单捆处理+合计”的方式，需要对捆间进行分割或者对最终结果进行去重，难度较大。

## 2.数据准备

数据集中包含了250张已经标注好的数据，具体数据源于datafountain和广联达公司举办的钢筋计数比赛，用户可直接访问[链接](https://www.datafountain.cn/competitions/332/datasets)下载

更多数据格式信息请参考[数据标注说明文档](https://paddlex.readthedocs.io/zh_CN/develop/data/annotation/index.html)

- **数据切分** 将训练集、验证集按照8.5:1.5的比例划分。 PaddleX中提供了简单易用的API，方便用户直接使用进行数据划分。

```
paddlex --split_dataset --format VOC --dataset_dir dataset --val_value 0.15
```

数据文件夹切分前后的状态如下:

```
  dataset/                          dataset/
  ├── Annotations/      -->         ├── Annotations/
  ├── JPEGImages/                   ├── JPEGImages/
                                    ├── labels.txt
                                    ├── train_list.txt
                                    ├── val_list.txt
```

## 3.模型选择

PaddleX提供了丰富的视觉模型，在目标检测中提供了RCNN和YOLO系列模型。在本项目中采用YOLO系列作为检测模型进行钢筋计数。

## 4. 模型训练

在项目中，我们采用yolov3作为钢筋检测的模型。具体代码请参考[train.py](./code/train.py)。

运行如下代码开始训练模型：

```
python code/train.py
```

若输入如下代码，则可在log文件中查看训练日志，log文件保存在`code`目标下

```
python code/train.py > log
```

- 训练过程说明

<div align="center">
    <img src="./images/3.png" width="1024"/>
</div>



## 5.模型优化(进阶)

- 精度提升 为了进一步提升模型的精度，在项目中采用了一系列模型优化的方式，具体请参考[模型优化文档](./accuracy_improvement.md)

采用PaddleX在Tesla V100上测试模型的推理时间（输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU的时间），推理时间如下表所示：（**20次推理取平均耗时**）

> 由于参与推理的图片较大，因此预处理时间在**Resize处**会消耗较多的时间。

| 模型                                                         | 推理时间 （ms/image） | map(Iou-0.5) | (coco)mmap |
| ------------------------------------------------------------ | :-------------------: | ------------ | ---------- |
| **baseline: YOLOv3 - MobileNetV1+label_smooth=False +img_size(480)** |      **114.15**       | 65.3         | 38.3       |
| YOLOv3 + MobileNetV1 + label_smooth=True +img_size(480) |        114.30         | 63.2         | 29.1       |
| YOLOv3 + MobileNetV1 + label_smooth=True + cluster_yolo_anchor+img_size(480) |        114.26         | 57.4         | 29.0       |
| YOLOv3 + ResNet34 + label_smooth=False + img_size(480) |        116.06         | 67.8         | 45.4       |
| YOLOv3 + ResNet34 + label_smooth=False  +cluster_yolo_anchor+ img_size(480) |        117.12         | 66.5         | 37.7       |
| YOLOv3 + ResNet34 + label_smooth=False + img_size(608)   |        117.82         | 68.4     | 47.9   |
| YOLOv3 + ResNet50_vd_ssld + label_smooth=False + img_size(480) |        120.16         | 67.9         | 48.4       |
| **YOLOv3 + ResNet50_vd_ssld + label_smooth=False + img_size(608)** |        120.44         | **69.0**         | **49.7**       |
| YOLOv3 + DarkNet53 + label_smooth=False+ img_size(480)       |        119.24         | 67.7         | 46.2       |
| YOLOv3 + DarkNet53 + label_smooth=False+ img_size(608)       |        120.05         | 67.9         | 47.1       |


注意:

- **608的图像大小**，一般使用默认的anchors进行训练和推理即可。
- **cluster_yolo_anchor**: 来源**paddlex2.0API**，用于生成拟合数据集的模型anchor

```
anchors = train_dataset.cluster_yolo_anchor(num_anchors=9, image_size=480)
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
```



**优化进展说明**：

- 1.通过选择**更好的backbone**作为特征提取的骨干网络可以提高识别率、降低漏检率。<**MobileNetV1 到 ResNet50_vd_ssld**>

- 2.通过选择理论上更好的检测架构**PPYOLO**未获得明显的完全性能提升(`Iou0.5提升不是很大`)，但也提高了检测的mmap值——即**Neck，Head部分的优化**可以提高ap。<**YOLOV3 到 PPYOLO**>

- 3.缩放适当的图像大小可以**提高模型的识别率**，但是存在一定的阈值——当图像大小到某一个阈值时会导致精度下降。

  - **一般图像大小选择(YOLO系列)**:320，480， 608。
  - 一般**图像如果较大，物体也比较大**，可以较为放心的缩小图像大小再进行相关的训练和预测。
  - 物体较小，不易缩小，**可以适当的裁剪划分原图或放大**，并处理对应的标注数据，再进行训练。

  <**480到608**>

- 4.一般情况通过cluster_yolo_anchor生成当前网络输入图像大小下拟合数据集的预置anchors，利用新生成的anchors替换原来的默认anchor，会使得模型预测定位上框选**位置更准确**，但是本次实验并未得到提升。

- 5.最初的两个YOLOV3的实验可以发现，本次优化中，如果**开启label_smooth，会导致预测ap下降**——原因是**标签平滑相当于一种正则化**，当前模型存在的问题是拟合不够好，因此加入标签平滑会适得其反。

- 通过以上的简单优化方式，获取了两个较好的模型结果:【前者模型**体积更小**】

- | 模型                                                         | 推理时间 （ms/image） | map(Iou-0.5) | (coco)mmap |
  | ------------------------------------------------------------ | --------------------- | ------------ | ---------- |
  | **baseline: YOLOv3 - MobileNetV1+label_smooth=False +img_size(480)** | **114.15**            | 65.3         | 38.3       |
  | **YOLOv3 + ResNet50_vd_ssld + label_smooth=False + img_size(608)** |        120.44         | **69.0**         | **49.7**       |

## 6.模型预测

运行如下代码：

```
python code/infer.py
```

则可生成result.txt文件并显示预测结果图片，result.txt文件中会显示图片中每个检测框的位置、类别及置信度，并给出检测框的总个数，从而实现了钢筋自动计数。

预测结果如下：

<div align="center">
    <img src="./images/4.jpg" width="800"/>
</div>
# 7.模型导出

模型训练后保存在output文件夹，如果要使用PaddleInference进行部署需要导出成静态图的模型,运行如下命令，会自动在output文件夹下创建一个`inference_model`的文件夹，用来存放导出后的模型。

```
paddlex --export_inference --model_dir=output/yolov3_ResNet50_vd_ssld/best_model --save_dir=output/inference_model --fixed_input_shape=[608,608]
```

**注意**：设定 fixed_input_shape 的数值需与 eval_transforms 中设置的 target_size 数值上保持一致。

# 8.模型上线选择

## (1)针对GPU端部署

本案例选择**面向GPU端**的最终方案是选择一阶段检测模型**YOLOV3**，其骨干网络选择加入了**ResNet50_vd_ssld**，训练阶段数据增强策略采用RandomHorizontalFlip、RandomDistort、RandomCrop等。

在Tesla V100的Linux系统下，模型的推理时间大约为**120.44ms/image**，包括**transform、输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU**的时间。

| 模型                                                       | 推理时间 (ms/image) | map(Iou-0.5) | (coco)mmap |
| ---------------------------------------------------------- | ------------------- | ------------ | ---------- |
| **YOLOv3 + ResNet50_vd_ssld + label_smooth=False + img_size(608)** |        120.44         | **69.0**         | **49.7**       |

**上线模型的PR曲线:**

<div align="center">
    <img src="./images/5.png" width="1024"/>
</div>



在本项目中的钢筋检测数据中，标注信息错误较少，但是**拍摄角度以及光照变化范围较大**，从而影响模型的学习。针对本项目的问题而言，钢筋检测中的**召回率比较低，而准确率较高**。

**注意**: 通过COCO指标使得评估时对多Iou尺度进行一个较完整的评估，使得得到该评估指标下最好的模型，以期望获得最好的泛化能力。

## (2)针对移动端部署

本案例选择**面向移动端**的最终方案是选择一阶段检测模型**YOLOV3**，其骨干网络选择加入了**MobileNetV1**，训练阶段数据增强策略采用RandomHorizontalFlip、RandomDistort、RandomCrop等。

在Tesla V100的Linux系统下，模型的推理时间大约为**114.15ms/image**，包括**transform、输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU**的时间。

| 模型                                                         | 推理时间 (ms/image) | map(Iou-0.5) | (coco)mmap |
| ------------------------------------------------------------ | ------------------- | ------------ | ---------- |
| **baseline: YOLOv3 - MobileNetV1+label_smooth=False +img_size(480)** | **114.15**          | 65.3         | 38.3       |

**上线模型的PR曲线:**

<div align="center">
    <img src="./images/6.png" width="1024"/>
</div>

相对于GPU端上线模型，该模型具有以下优势:

- 模型参数大小仅为: **96.8MB**，而GPU端模型为:**181MB**。
- 输入图像更小，模型计算**时间花销更小**。
- **召回率更低**，可能有更少的误检——但由于模型轻量化缘故，漏检情况可能会更多一点。

## 9.模型部署方式
模型部署采用了PaddleX提供的C++ inference部署方案，在方案中提供了基于Paddle Lite部署[Demo](https://github.com/yzl19940819/example_Demo/tree/master/android_demo)，用户可根据实际情况自行参考。

<div align="center">
    <img src="./images/android.png" width="300"/>
</div>

* 感谢[蔡敬辉](https://github.com/cjh3020889729)同学对本案例的贡献
* 示例图片引用说明：案例中钢筋数据样本源于datafountain和广联达公司举办的钢筋计数比赛，用户可直接访问[链接](https://www.datafountain.cn/competitions/332/datasets)

