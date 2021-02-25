[English](README_en.md) | 简体中文

# 手写数字识别案例

## 依赖模块
* os
* json
* gzip
* numpy
* random
* time
* paddlepaddle==2.0.0

## 项目介绍
```
|-datasets: 存放数据和数据读取脚本
    |-mnist.json.gz: mnist数据文件
    |-generate.py: 读取数据的脚本
|-nets: 存放网络结构
    |-fnn.py: 该脚本中定义了单层和多层两种前馈神经网络
|-train.py: 启动训练的脚本
```

## 实验结果
|模型结构  |激活函数  |正则化     |optimizer|epoch  |lr    |bs    |acc   |
|:--:     |:--:    |:--:     |:--:     |:--:   |:--:  |:--:  |:--:  |
|单层前馈神经网络  |sigmoid  | N       |SGD      |10     |0.1   |32    |85.03%|
|单层前馈神经网络  |sigmoid  | N       |SGD      |10     |1     |32    |95.87%|
|单层前馈神经网络  |relu     | N       |SGD      |10     |0.1   |32    |96.18%|
|多层前馈神经网络  |relu     | N       |SGD      |10     |0.1   |32    |97.10%|
|多层前馈神经网络  |relu     | Y       |SGD      |10     |0.1   |32    |97.18%|


## 训练
直接使用 train.py 脚本启动训练。
```
python3 train.py
```