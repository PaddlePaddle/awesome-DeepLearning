# 情感分析  [[English](./README_en.md)]

## 依赖模块
- re
- os
- numpy
- paddle
- random
- tarfile
- requests

其中paddle请安装2.0版本，具体安装方式请参考
  [飞桨官网->快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html) 。

## 目录结构
```buildoutcfg
|-data: 存放数据集或模型
|-model: 模型相关组件
    |-sentiment_classifier: 情感分析模型的实现
|-utils: 存放工具式接口或函数
    |-data_processor.py: 数据处理相关的操作
|-train.py: 启动训练的脚本
|-evaluate.py: 启动测试的脚本
```

## 启动训练
>python train.py

## 启动评估
>python evaluate.py