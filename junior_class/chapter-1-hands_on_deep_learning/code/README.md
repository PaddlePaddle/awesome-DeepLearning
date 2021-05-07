# **波士顿房价预测**

## **依赖模块**

- numpy
- os
- random
- matplotlib
- paddlepaddle==2.0.0

## **项目介绍**

```buildoutcfg 
|-work: 存放波士顿房价预测数据集 
|-1-2-build_neural_network_using_numpy.py: 使用Python语言和Numpy库构建神经网络模型的脚本 
|-1-4-build_neural_network_using_paddle.py: 使用飞桨实现房价预测模型的脚本
```

## **数据集准备**
    
下载[housing.data](https://aistudio.baidu.com/aistudio/datasetdetail/58711)数据集到work目录下
        
## **训练**

1. 使用Python语言和Numpy库实现房价预测模型训练''' python3  1-2-build_neural_network_using_numpy.py'''
2. 使用飞桨实现房价预测模型训练''' python3 1-4-build_neural_network_using_paddle.py '''