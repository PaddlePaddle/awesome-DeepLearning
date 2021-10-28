# 前言

本项目为百度论文复现第四期《Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields ∗》论文复现第一名代码以及模型。 依赖环境：

* paddlepaddle-gpu2.1.2以上
* python3.7

代码在MPII数据集上表现效果很好，精度复现也很高。

# 介绍

传统的自顶向下的姿态估计算法都是采用先定位到人，再做单人关键点检测。这种方法的缺点在于：

图像中人的数量、位置、尺度大小都是未知的，人与人之间的交互遮挡可能会影响检测效果，最最重要的是，运行时间复杂度随着图像中个体数量的增加而增长，无法实时检测。
利用人体姿态估计，判断人体关键点位置，从而根据关键点判断人体姿态。

本文的重点在于提出PAFs（Part Affinity Fields，怎么翻译？部分亲和场？部分关联域？），翻译起来很别扭，看一下图就理解了：

![](https://ai-studio-static-online.cdn.bcebos.com/93859004c3ac4374811bfcb61f52fdcf56c89c99c82246589581b3307d1c59b2)

参考论文：
* Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh (2016). Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields. CoRR, abs/1611.08050.


参考项目
* [https://github.com/MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
* [https://github.com/bearpaw/pytorch-pose/tree/master/evaluation](https://github.com/bearpaw/pytorch-pose/tree/master/evaluation)
* [https://github.com/NieXC/pytorch-ppn](https://github.com/NieXC/pytorch-ppn)
* [https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)
* [https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)
* [https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation)
* [https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation](https://github.com/dragonfly90/mxnet_Realtime_Multi-Person_Pose_Estimation)

项目AI Studio地址
* [https://aistudio.baidu.com/aistudio/projectdetail/2306743](https://aistudio.baidu.com/aistudio/projectdetail/2306743)

# 对比
本项目基于paddlepaddle深度学习框架复现，对比于作者论文的代码：

* 我们将提供更加细节的训练流程，以及更加全面的预训练模型。
* 我们将提供基于AI Studio平台可在线运行的项目地址，您不需要在您的机器上配置paddle环境可以直接浏览器在线运行全套流程。
* 我们的提供模型在MPII数据集上超越原作者论文最好模型4%左右。

# 模型整体流程

## 首要条件
```python
export PYTHONPATH="$PWD":$PYTHONPATH  #终端执行
```

## 训练
可通过`training/config.yml` 文件夹修改训练超参数
```python
python training/train_pose.py --config ./trainning/config.yml --train_dir ./datasets/process_train.json --val_dir ./datasets/process_val.json
```
## 评估
* `model`：模型权重
* `only_eval`: 已有评估完成存在的predection.npy文件，对此评估。

```python
python testing/eval.py  --model ./RMPose_PAFs.pdparams.tar -only_eval True
```

## 测试
* `image_dir`: 支持文件夹路径，以及单文件处理
* `model`: 模型权重
* `output`: 输出文件夹路径
```python
python testing/test_pose.py --image_dir ./ski.jpg --model ./RMPose_PAFs.pdparams.tar
```

# 评估指标
参考代码（Matlab版）： [https://github.com/anibali/eval-mpii-pose](https://github.com/anibali/eval-mpii-pose)

MPII数据集的评估指标采用的是PCKh@0.5。预测的关节点与其对应的真实关节点之间的归一化距离小于设定阈值，则认为关节点被正确预测，PCK即通过这种方法正确预测的关节点比例。

PCK@0.2表示以躯干直径作为参考，如果归一化后的距离大于阈值0.2，则认为预测正确。

PCKh@0.5表示以头部长度作为参考，如果归一化后的距离大于阈值0.5，则认为预测正确。

在本项目中论文复现结果：

|Method         | Head    |Shoulder | Elbow   |  Wrist   | Hip   |Knee   |Ankle  |Mean   |
| --------       | -------- | -------- | -------- | -------- |--------|--------|--------|--------|
| **RMPose_PAFs**| 36.94    |92.19   | 86.21   | 79.77   |85.62   |81.62  | 76.64  | 80.96 |
|**原论文**     |91.2     |887.6   |77.7    |66.8     |75.4    |68.9   |61.7   |75.6|
|**DeeperCut**| 73.4       | 71.8   | 57.9   | 39.9    | 56.7    | 44.0 | 32.0   | 54.1 |
|**AlphaPose** | 91.3	    |90.5    |	84.0	|76.4	   | 80.3	|79.9	|72.4	|82.1 |


# -----------------------分割线------------------
以下为运行部分(复制到终端运行) 注：**首先**在终端执行`export PYTHONPATH="$PWD":$PYTHONPATH`

## 1. 数据的加载与处理


```python
!mkdir /home/aistudio/data/mpii
%cd /home/aistudio/data/mpii/
!tar -zxvf /home/aistudio/data/data58767/mpii_human_pose_v1.tar.gz
!unzip /home/aistudio/data/data58767/annot.zip -d /home/aistudio/data/mpii/
%cd /home/aistudio/work/
!python datasets/preprocess_json.py
```

## 2. 模型的训练


```python
!python training/train_pose.py --config ./trainning/config.yml --train_dir ./datasets/process_train.json --val_dir ./datasets/process_val.json
```

## 3. 模型的预测


```python
!python testing/test_pose.py --image_dir ./ski.jpg --model ./RMPose_PAFs.pdparams.tar
```

预测结果如下：

![](https://ai-studio-static-online.cdn.bcebos.com/32760e242fae4a47a63b652d9de922e9b73ec8c607b54bf7986012be6c4889d1)


## 4.模型的评估与测试


```python
!python testing/eval.py  --model ./RMPose_PAFs.pdparams.tar -only_eval True
```

## 个人简介

> 江苏科技大学人工智能专业19级 芦星宇

> 发布多个CV竞赛Baseline
