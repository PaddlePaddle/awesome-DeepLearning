# 使用方法

1.https://aistudio.baidu.com/aistudio/competition/detail/78

去这里下载数据集，解压。将train_50k_mask重命名为mask，将train_image重命名为image

2.运行utils的py文件，需要填的路径为mask所在的路径

因为是四个模型，所以下面的训练和测试需要在四个模型上进行。
## 训练
训练需要运行train.py，参数配置看内部的函数config，每个参数都有对应的解释
，数据路径为参数train_dataset。填写image所在路径即可。

## 测试
训练可以在test.py文件中进行。按照参数提示输入即可

## AI Studio项目
如果觉得自己调试运行代码有麻烦，可在AI Studio中fork我的项目直接一键运行，项目链接：
https://aistudio.baidu.com/aistudio/projectdetail/2210815
