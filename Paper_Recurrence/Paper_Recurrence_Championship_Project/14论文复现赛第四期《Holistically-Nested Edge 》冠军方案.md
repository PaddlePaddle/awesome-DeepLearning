# **论文复现赛第四期《Holistically-Nested Edge 》冠军方案**
# 1.简介
本项目基于PaddlePaddle复现《 Holistically-Nested Edge Detection》论文。改论文提出了一种使用卷积神经网络来检测边缘的方法。并超越原文精度，达到ODS=0.787。


边缘是图像最基础也最重要的基本特征之一,边缘检测更是图像分割,模式识别等图像技术的重要前提.因此图像的边缘检测一直是图像处理领域研究的热点.一个好的边缘检测方法可以极大的提高图像处理的效率和正确率.

传统的Sobel滤波器，Canny检测器具有广泛的应用，但是这些检测器只考虑到局部的急剧变化，特别是颜色、亮度等的急剧变化，通过这些特征来找边缘，但这些特征很难模拟较为复杂的场景。而在HED方法中，一改之前边缘检测方法基于局部策略的方式，而是采用全局的图像到图像的处理方式。即不再针对一个个patch进行操作，而是对整幅图像进行操作，为高层级信息的获取提供了便利。与此同时，该方法使用了multi-scale 和multi-level, 通过groundtruth的映射在卷积层侧边插入一个side output layer，在side output layer上进行deep supervision，将最终的结果和不同的层连接起来。


论文地址：

[https://arxiv.org/abs/1504.06375](https://arxiv.org/abs/1504.06375)

本项目github地址：

[https://github.com/txyugood/hed](https://github.com/txyugood/hed) 精度:ODS=0.787


参考项目:

[https://github.com/sniklaus/pytorch-hed](https://github.com/sniklaus/pytorch-heds) 精度:ODS=0.774

[https://github.com/s9xie/hed](https://github.com/s9xie/hed) 精度:ODS=0.782 (原文项目)

[https://github.com/zeakey/hed](https://github.com/zeakey/hed) 精度:ODS=0.779







# 2.数据集下载

HED-BSDS:

[https://aistudio.baidu.com/aistudio/datasetdetail/103495](https://aistudio.baidu.com/aistudio/datasetdetail/103495)

# 3.环境

PaddlePaddle == 2.1.2

python == 3.7

# 4.VGG预训练模型

模型下载地址：

链接: [https://pan.baidu.com/s/1etmgEGtbhwxMECwIRkL1Lg](https://pan.baidu.com/s/1etmgEGtbhwxMECwIRkL1Lg)

密码: uo0e



# 5.训练

本论文发布时间较早，应该是比较早使用卷积神经网络来做图像边缘检测的项目。论文中使用了VGG16网络做为backbone。然后分别从VGG的5个部分做了5个分支，输出不同的边缘图，最后用一个卷积将5个边缘图融合，作为最终结果。

训练策略方面，文中将VGG的前4部分学习率的倍率设置为1，第五部分的学习率倍率是前四部分的100倍。这意味着，前四部分主要进行微调，第五部分需要重新学习，主要用来输出边缘图像。同时最后融合部分的卷积层的学习率的倍率设置为0.001，同时初始化值为0.2，这意味着，用较低的学习率来微调融合权重。

在训练过程中，论文使用SDG优化器，使用StepDecay动态调整学习率，学习率为1e-6。但是通过大量测试，效果并不理想，所以在本项目中将学习率设置为1e-4,同时使用Warmup和PolynomialDecay的方式动态的调整学习率，总迭代次数为100000次。最终评测结果0.787，超过了原文精度以及其他pytoch和caffe版本的复现项目。

下面开始启动训练，首先解压数据集。


```python
!tar xvf data/data102948/HED-BSDS.tar
```

然后运行训练脚本。


```python
%cd /home/aistudio/hed/
!python train.py --iters 100000 --batch_size 10 --learning_rate 0.0001 --save_interval 1000 --pretrained_model /home/aistudio/vgg16.pdparams --dataset /home/aistudio/HED-BSDS
```

# 6.测试


```python
!python predict.py --pretrained_model model_hed.pdparams --dataset /home/aistudio/HED-BSDS/test/ --save_dir output/result
```


上述命令中pretrained_model为训练结果模型，dataset为测试图片路径。

训练结果模型下载地址：

链接: [https://pan.baidu.com/s/1VXnrHCu9Wb7zAiOTsb0vFw](https://pan.baidu.com/s/1VXnrHCu9Wb7zAiOTsb0vFw)

密码: pocu

# 7.验证模型

预测结果需要使用另外一个项目进行评估。

评估项目地址:

[https://github.com/zeakey/edgeval](https://github.com/zeakey/edgeval)

运行环境 Matlab 2014a


本项目评估结果：

![](https://ai-studio-static-online.cdn.bcebos.com/21832c7c919247b6a7367dda2c78d35d1cc1185ddcf64b308f392a25bfe898e5)


推理预测结果：

![](https://ai-studio-static-online.cdn.bcebos.com/ccd1743220b7483695998a9741df788bf6eca047cf55470ba2a7c0ba4e427894)


![](https://ai-studio-static-online.cdn.bcebos.com/692205801ef34979a61425fb092ef92cdb66bcefbfd645a4bb3c74210f4e74b7)



# 8.总结
本篇论文是我本次论文复现赛选择的第一篇论文，这篇论文的代码比较简单，但是复也遇到一些问题。对比ResNet50这种网络，VGG16更难训练，即使使用了论文里推荐的参数也很难获取好的结果，所以就需要自己不断的去调整超参数。同时评测程序是运行在matlab环境下的，运行一次大概需要3个小时，这也大大的增加了复现所需要的时间。最后通过多次调参与采用了一些学习率衰减的策略，达到了要求精度，本篇论文复现也学到了很多东西，最后感谢百度提供这次比赛。
