# 眼疾识别 [[English](./README_en.md)]

## 依赖模块
- os
- numpy
- matplotlib
- PIL
- random
- paddlepaddle==2.0.0
- opencv

## 项目介绍
'''
|-data: 存放眼疾识别数据集
|-datasets: 存放数据读取脚本
    |-dataset.py: 读取数据脚本
|-CNN_Basis: 卷积神经网络基础模块的实现脚本
    |-Average_filtering.py: 均值滤波脚本
    |-BatchNorm1D.py: 1维BatchNorm脚本
    |-BatchNorm2D.py: 2维BatchNorm脚本
    |-Black_and_white_boundary_detection.py: 黑白边界检测脚本
    |-Dropout.py: Dropout脚本
    |-Edge_detection.py: 边缘检测脚本
    |-SimpleNet.py: 简单网络的实现脚本
|-nets: 存放网络定义脚本
    |-AlexNet.py: AlexNet网络定义脚本
    |-GoogLeNet.py: GoogLeNet网络定义脚本
    |-LeNet_PALM.py: LeNet网络定义脚本，输入图片为3通道
    |-LeNet.py: LeNet网络定义脚本，输入图片为单通道
    |-ResNet.py: ResNet网络定义脚本
    |-vgg.py: VGG网络定义脚本
|-Paddle_highAPI.py: 体验Paddle高层API的脚本
|-train_MNIST.py: 使用MNIST数据集训练LeNet的脚本
|-train_PALM.py: 训练眼疾识别模型的脚本

'''

## 数据集准备
1. 下载[眼疾识别数据集](https://aistudio.baidu.com/aistudio/datasetdetail/19065)到data目录下
2. 解压数据集
‘’‘
cd data
!unzip -o -q training.zip
%cd palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q ../validation.zip
!unzip -o -q ../valid_gt.zip
#返回code目录
%cd ../../
’‘’

## 训练
1. 使用MNIST数据集训练LeNet
'''
python3 train_MNIST.py
'''
2. 训练眼疾识别模型
'''
python3 train_PALM.py
'''
