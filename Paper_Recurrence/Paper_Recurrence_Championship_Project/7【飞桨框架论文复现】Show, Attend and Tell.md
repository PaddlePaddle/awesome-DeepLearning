# 前言

本项目为百度论文复现赛第四期《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》论文复现代码。

依赖环境：

- paddlepaddle-gpu2.1.2
- python3.7

代码在coco2014数据集上训练，复现精度：

|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.721 |0.547|0.405|0.300| 0.242|0.525|0.924|


# 模型背景及其介绍

参考论文：《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》[论文链接](https://dl.acm.org/doi/10.5555/3045118.3045336)

近年来，人们提出了几种生成图像描述生成方法。这些方法中许多都是基于递归神经网络，并受到了成功使用序列与神经网络进行机器翻译训练的启发。图像描述生成非常适合机器翻译的编码器-解码器框架，一个主要原因是它类似于将图像翻译成句子。

受机器翻译和目标检测工作的启发，论文首次提出在图像描述模型中引入注意力机制，大幅度提高了模型的性能，并可视化展示了注意力机制如何学习将目光固定在图像的显著目标上，整体框架如下。

![](https://ai-studio-static-online.cdn.bcebos.com/cfd71f6849c5460e9ff8a1079cebb7a29557b4e4c4394f06ac350b31a7e7d7e4)

第一步：输入Image到模型中。

第二步：经过CNN进行卷积提取Image特征信息最终形成Image的特征图信息。

第三步：attention对提取的特征图进行加权求和，作为后续进入LSTM模型的输入数据，不同时刻的attention数据会受到上一时刻状态输出数据的影响。

第四步：LSTM模型最终输出caption。

模型结构：
![](https://ai-studio-static-online.cdn.bcebos.com/5e14f9090a7549818ac5d65be61cb0fbd9710558a57a4282a51c088de3b67934)

[参考项目地址链接](https://github.com/ruotianluo/ImageCaptioning.pytorch)

[复现论文代码github地址链接](https://github.com/chenlizhi-1013/paddle-show-attend-and-tell-captioning)

# 数据集

coco2014 image captions [论文](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)，采用“Karpathy” data split [论文](https://arxiv.org/pdf/1412.2306v2.pdf)

数据集总大小：123287张

- 训练集：113287张

- 验证集：5000张

- 测试集：5000张

标签文件：dataset_coco.json

# 运行

## 解压预训练数据到work/data/目录下
预训练数据包括: 通过vgg19提取的coco2014图像网格特征、cocotalk.json、cocotalk_label.h5

通过命令 !python3 scripts/prepro_feats.py 和 !python3 scripts/prepro_labels.py 获得


```python
%cd /home/aistudio/work/data/
!unzip -oq /home/aistudio/data/data106948/coco_data_vgg.zip
```

## 解压用于训练测试的文件到work/目录下


```python
%cd /home/aistudio/work/
!unzip -oq /home/aistudio/data/data107076/coco-caption.zip
```

## 安装依赖库


```python
%cd /home/aistudio/work/
!pip install -r requirements.txt
```

## 训练

训练的日志和模型会放到work/log/目录下


```python
!python3 train.py
```

## 评估

我已经将训练好的model_best.pdparams文件放在了work/log目录下

加载work/log目录下保存的训练模型数据进行验证


```python
%cd /home/aistudio/work/
!unzip -oq /home/aistudio/data/data107076/log.zip
```


```python
!python3 eval.py
```
