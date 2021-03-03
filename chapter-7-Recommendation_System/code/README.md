# 电影推荐系统 [[English](./README_en.md)]

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
```
|-data: 存放movielens识虫数据集
|-net: 存放网络定义脚本
    |-DSSM.py: 该脚本中定义了movielens的网络结构
|-train.py: 启动训练和验证的脚本

```

## 数据集准备
1. 下载[movielens数据集](https://aistudio.baidu.com/aistudio/datasetdetail/3233)到data目录下
2. 解压数据集
‘’‘
cd data
unzip -q ml-1m.zip
’‘’

## 训练
直接使用 train.py 脚本启动训练。
'''
python3 train.py
'''
