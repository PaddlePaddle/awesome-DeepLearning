# Eye disease recognition [[简体中文](./README.md)]

## Dependent packages
- os
- numpy
- matplotlib
- PIL
- random
- paddlepaddle==2.0.0
- opencv

## Structure
'''
|-data: store dataset
|-datasets: script to store data reading
    |-dataset.py: script to read data
|-CNN_Basis: implementation script of basic module of convolutional neural network
    |-Average_filtering.py: mean filter script
    |-BatchNorm1D.py: 1 channel BatchNorm script
    |-BatchNorm2D.py: 2 channels BatchNorm script
    |-Black_and_white_boundary_detection.py: Black and white border detection script
    |-Dropout.py: Dropout script
    |-Edge_detection.py: Edge detection script
    |-SimpleNet.py: Simple network implementation script
|-nets: 存放网络定义脚本
    |-AlexNet.py: AlexNet network definition script
    |-GoogLeNet.py: GoogLeNet network definition script
    |-LeNet_PALM.py: LeNet network definition script，Input picture is 3 channels
    |-LeNet.py: LeNet network definition script，Input picture is 1 channel
    |-ResNet.py: ResNet network definition script
    |-vgg.py: VGG network definition script
|-Paddle_highAPI.py: experience the script of Paddle high-level API
|-train_MNIST.py: script for training LeNet using MNIST dataset
|-train_PALM.py: script for training eye disease recognition model

'''

## Dataset preparation
1. Download the [dataset](https://aistudio.baidu.com/aistudio/datasetdetail/19065) to the data directory
2. Unzip the dataset
‘’‘
cd data
!unzip -o -q training.zip
%cd palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q ../validation.zip
!unzip -o -q ../valid_gt.zip
%cd ../../
’‘’

## Train
1. Train LeNet using MNIST dataset
'''
python3 train_MNIST.py
'''
2. Training eye disease recognition model
'''
python3 train_PALM.py
'''
