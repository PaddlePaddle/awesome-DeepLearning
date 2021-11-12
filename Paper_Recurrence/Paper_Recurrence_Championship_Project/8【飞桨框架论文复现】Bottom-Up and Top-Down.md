# 前言

本项目为百度论文复现赛第四期《Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering》论文复现代码。

依赖环境：

- paddlepaddle-gpu2.1.2
- python3.7

代码在coco2014数据集上训练，复现精度：

Cross-entropy Training

|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.761 |0.598|0.459|0.350| 0.272|0.562|1.107|0.203|

SCST(Self-critical Sequence Training)

|Bleu_1|Bleu_2|Bleu_3|Bleu_4|METEOR|ROUGE_L|CIDEr|SPICE|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.799 |0.641|0.493|0.373| 0.275|0.580|1.202|0.209|

# 论文背景及其介绍
参考论文：《Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering》[论文链接](https://ieeexplore.ieee.org/document/8578734)

在人类视觉系统中，存在自上而下(Top-Down Attention)和自下而上(Bottom-Up Attention)两种注意机制。前者注意力由当前任务所决定，我们会根据当前任务聚焦于与任务紧密相关的部分，后者注意力指的是我们会被显著的、突出的事物所吸引。
视觉注意大部分属于自上而下类型，图像作为输入，建模注意权值分布，然后作用于CNN提取的图像特征。然而，这种方法的注意作用图像对应于下图的左图，没有考虑图片的内容。对于人类来说，注意力会更加集中在图片的目标或其他显著区域，所以论文作者引进自下而上注意(Bottom-Up Attention)机制，如下图的右图所示，注意力作用于显著物体上。
![](https://ai-studio-static-online.cdn.bcebos.com/22f6f96399604e24810f02983ec375c2cd7260e043eb46629c4766b48b7680f8)

Caption model结构如下图所示，模型共有2个LSTM模块，一个是Language LSTM，另一个是Top-Down Attention LSTM。
本文的Bottom-Up Attention 用的是目标检测(object detection)领域的Faster R-CNN方法来提取。

![](https://ai-studio-static-online.cdn.bcebos.com/d14527e84dbd40f5be6e2e8759f869cb9cceef6e566c4249957ac3188ad54ad1)

[参考项目地址链接](https://github.com/ruotianluo/ImageCaptioning.pytorch)

[复现论文代码github地址链接](https://github.com/chenlizhi-1013/paddle-bottom-up-attention-captioning)

# 数据集
coco2014 image captions [论文](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48)，采用“Karpathy” data split [论文](https://arxiv.org/pdf/1412.2306v2.pdf)

数据集总大小：123287张

训练集：113287张

验证集：5000张

测试集：5000张

标签文件：dataset_coco.json


# 运行

## 解压预训练数据到work/data/目录下
预加载数据包括: 通过Faster R-CNN提取的coco2014图像显著区域特征（cocobu_att）、池化特征（cocobu_fc）、边框特征（cocobu_box）；
cocotalk.json；cocotalk_label.h5。

上述预训练数据也可以通过命令 !python3 scripts/make_bu_data.py 和 !python3 scripts/prepro_labels.py 获得

显著区域特征（cocobu_att）因数据过大，原数据分成了cocobu_att_train和cocobu_att_val上传


```python
%cd /home/aistudio/work/data/
!unzip -oq /home/aistudio/data/data107198/cocobu_att_train.zip
!unzip -oq /home/aistudio/data/data107198/cocobu_att_val.zip
!unzip -oq /home/aistudio/data/data107198/cocobu_fc.zip
!unzip -oq /home/aistudio/data/data107198/cocobu_box.zip
```

加载完成后，我们把cocobu_att_train和cocobu_att_val合并成cocobu_att


```python
%cd /home/aistudio/work/data/
!mv cocobu_att_val/* cocobu_att_train/
!mv cocobu_att_train cocobu_att
!find . -type d -empty -delete
```

## 解压用于训练测试的文件coco-caption到work/目录下


```python
%cd /home/aistudio/work/
!unzip -oq /home/aistudio/data/data108181/coco-caption.zip
```

## 安装依赖库


```python
%cd /home/aistudio/work/
!pip install -r requirements.txt
```

## 训练
训练的日志和模型会放到work/log/目录下

训练过程过程分为两步：Cross-entropy Training和SCST(Self-critical Sequence Training)


```python
# Cross-entropy Training
!python3 train.py --cfg configs/updown.yml
```


```python
# SCST(Self-critical Sequence Training)
!python3 train.py --cfg configs/updown_rl.yml
```

## 评估
解压预先训练好的模型日志log到work/目录下

加载work/log目录下保存的训练模型数据进行验证


```python
%cd /home/aistudio/work/
!unzip -oq /home/aistudio/data/data108181/log.zip
```


```python
!python3 eval.py
```
