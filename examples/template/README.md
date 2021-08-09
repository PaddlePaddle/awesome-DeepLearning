# examples



## 作业简介

作业包括深度学习基础知识和代码实践，比如下面的形式：

### 1. 深度学习基础知识 

+ CNN-DSSM知识点补充：补充DSSM变体模型的知识点，主要包括：概念，模型，作用，场景，优缺点等。 （10分）
+ LSTM-DSSM知识点补充：补充DSSM变体模型的知识点，主要包括：概念，模型，作用，场景，优缺点等。
+ MMoE多任务学习知识点补充：补充主流的推荐模型MMoE模型的知识点，主要包括：概念，模型，作用，场景，优缺点等。   
+ ShareBottom多任务学习知识点补充：补充经典的多任务学习ShareBottom模型的知识点，主要包括：概念，模型，作用，场景，优缺点等。   
+ YouTube深度学习视频推荐系统知识点：补充视频推荐的经典架构知识点，主要包括：概念，流程，原理，作用，优缺点等。   

### 2. 代码实践

写一个广告推荐领域的“点击率预估模型”案例：

+ 可选网络结构：DNN,DeepFM、wide&deep；

+ 广告点击数据集Criteo网址： https://www.kaggle.com/c/criteo-display-ad-challenge/

+ 具体形式为理论+代码。

## 依赖模块

书写程序运行所需要的python及相应的库的版本，比如：

+ Python 3.x
+ paddlepaddle==2.1.1
......


## 项目结构说明

需要解释每一个python文件的作用和功能，必须包含的文件train.py, predict.py, 知识点的md文件，训练日志training.log， 比如下面的结构：

```
|-data_process.py: 数据预处理 
|-dataloader.py: 包含构建dataloader工具函数
|-model.py: 模型的定义
|-train.py: 训练文件，训练会输出日志等等
|-predict.py: 启动模型预测的脚本，并且储存预测结果于txt文件
|- example.md :  example知识点的md文件
|-example.ipynb : 模型的notebook文件
|-training.log :训练过程的日志文件
```



## 代码实践运行

需要把每一步的执行命令写清楚，并且有特殊的配置也需要相信说明，下面给出示例：

### 数据预处理

```
python data_process.py
```

### 模型训练与评估
```
python train.py
```

### 模型预测
```
python predict.py
```



