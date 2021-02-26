# 使用word2vec训练词向量
## 依赖模块
- requests==2.22.0  
- paddle==2.0.0

如未安装paddle，请点击
  [飞桨官网->快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/2.0/install/pip/windows-pip.html) 查看安装方式。
    
## 目录结构
```buildoutcfg
|-data: 存放下载后的数据
|-model: 模型相关组件
    |-word2vec: skip gram的模型实现
|-utils: 存放工具式接口或函数
    |-data_processor.py: 数据处理相关的操作
    |-utils: 一些工具式的函数
|-train.py: 启动训练的脚本
```

## 启动训练
>python train.py