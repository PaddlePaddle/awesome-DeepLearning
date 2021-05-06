# AI 识虫 [[English](./README_en.md)]

## 依赖模块
- os
- numpy
- xml
- opencv
- PIL
- random
- paddlepaddle==2.0.0
- time
- json

## 项目介绍
'''
|-data: 存放AI识虫数据集
|-datasets: 存放数据预处理和数据读取脚本
    |-transform.py: 数据预处理脚本
    |-dataset.py: 读取数据脚本
|-Detection_basis: 目标检测基础模块的实现脚本
    |-box_iou_xywh.py: 使用‘xywh’格式计算iou的脚本
    |-box_iou_xyxy.py: 使用‘xyxy’格式计算iou的脚本
    |-draw_anchor_box.py: 绘制锚框的脚本
    |-draw_rectangle.py: 绘制框体的脚本
    |-mAP.py: mAP计算脚本
    |-multiclass_nms.py: 多类别nms脚本
    |-nms.py: 单类别nms脚本
|-net: 存放网络定义脚本
    |-YOLOv3.py: 该脚本中定义了YOLOv3的网络结构以及损失函数定义
|-train.py: 启动训练的脚本
|-predict.py: 推理所有验证集图片的脚本
|-predict_one_pic.py: 推理一张图片的脚本

'''

## 数据集准备
1. 下载[AI识虫数据集](https://aistudio.baidu.com/aistudio/datasetdetail/19638)到data目录下
2. 解压数据集
‘’‘
cd data
unzip -q insects.zip
’‘’

## 训练
直接使用 train.py 脚本启动训练。
'''
python3 train.py
'''

## 推理
1. 使用验证集所有图片进行推理
'''
python3 predict.py
'''
2. 使用一张图片进行推理
'''
predict_one_pic.py
'''
