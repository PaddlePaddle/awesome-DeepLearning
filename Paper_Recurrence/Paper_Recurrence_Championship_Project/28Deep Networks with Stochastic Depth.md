# 使用PaddlePaddle复现论文：Deep Networks with Stochastic Depth
#### 基于Stochastic Depth的基于cifar10数据集的ResNet110模型
[Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382v3.pdf)

##  一·、简介
#### 摘要：
  具有数百层的非常深的卷积网络大大减少了在竞争性基准上的错误。尽管测试时许多层的无与伦比的表现力是非常理想的，但是训练非常深的网络也带来了它自己的一系列挑战。比如梯度可能会消失，前向传播的信息通常会减少，而且训练时间可能慢得令人痛苦。要解决这些问题，论文提出了Stochastic Depth，这是一种训练过程，使得训练短网络和在测试时使用深网络这看似矛盾的设置成功。论文作者从非常深的网络开始，但在训练过程中，对于每个小批量，随机删除模型所有层的一个子集，并用 identity function绕过这些层。这种简单的方法是对残差网络（resnet系列）的最近的成功进行补充。它大大减少了训练时间，并显著改善了论文作者使用的几乎所有数据集上的测试误差。利用随机深度，论文作者说可以增加深度，即使超过1200层的剩余网络，仍可提供有意义的改善的测试误差(CIFAR-10为4.91%)。

   本项目是基于Stochastic Depth的基于cifar10数据集的ResNet110模型在 Paddle 2.x上的开源实现。该模型有3个Layer，每个Layer分别由18个BasicBlock组成，每个BasicBlock由两个conv-bn-relu和skip connection组成，其中按论文在每个mini-batch进行按照论文公式计算出的linear_decay的各block的drop_rate(论文中是保留率，1-drop_rate)一次伯努利采样，根据采样的结果决定各block保不保留，由此在训练时减小了模型平均长度，加快了训练，且测试时用full depth有模型集成的效果，提高了精度。
   论文效果图：

   ![](https://ai-studio-static-online.cdn.bcebos.com/02dd2fa52fee448cb4636d448168a929b67778c909bc405895333e6723916a9c)
## 二、复现精度
本次比赛的验收标准： CIFAR-10 test error=5.25 （论文指标）。我们的复现结果对比如下所示：

<table>
    <thead>
        <tr>
            <th>来源</th>
            <th>test error</th>
            <th>test acc</th>
            <th>模型权重</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>原论文实现（指标）</td>
            <td>5.25%</td>
            <td>94.75%</td>
            <td>无</td>
        </tr>
        <tr>
            <td>pytorch复现（目录下已提供源码）</td>
            <td>4.97%</td>
            <td>95.03%</td>
            <td>https://pan.baidu.com/s/1d0PX45K73JHFeHc19X0eTA</td>
        </tr>
        <tr>
            <td>paddle实现</td>
            <td>5.23%</td>
            <td>94.77%</td>
            <td>https://pan.baidu.com/s/1Kzd_bQDVbHIL7J0-NZWkqw</td>
          </tr>
         <tr>
            <td>paddle实现（不按照论文训练、验证、测试划分，随机数种子2021，两次实验结果）</td>
            <td>5.12% 5.03%</td>
            <td>94.88% 94.97%</td>
            <td>https://pan.baidu.com/s/1w5YQhEyASPNPHTDPJFJswA https://pan.baidu.com/s/1YbnfoPQcNqPSh7cTyk3fhw</td>
        </tr>
    </tbody>
</table>
模型权重提取码：zpc6

##### 注意：我们完全按照论文的实现是 paddle实现 这一栏，最后一栏的实现只有数据集划分不一样（即传统的数据集划分（50000训练集、10000测试集，无验证集），这与原文其实不符）。至8月9号早上更新七、参考 是否选择官方实现或作者推荐的pytorch实现的代码（结果没跑完，校园网断了，离谱）

> #### Known Problems（来自作者所说，我们的代码也是这样）
>      It is normal to get a +/- 0.2% difference from our reported results on CIFAR-10, and analogously for the other datasets. Networks are initialized differently, and most importantly, the validation set is chosen at random (determined by your seed).

#### 训练loss图

![](https://ai-studio-static-online.cdn.bcebos.com/00885fc5a7404c4f9e356fca9a14de9f2a61806236bd4c63bab991c971431a0d)

#### val acc 图
由于test acc只在最后使用最好val acc的模型评估一次，故只能以val acc图代替（val_acc很高，最高95.7%）

![](https://ai-studio-static-online.cdn.bcebos.com/1ab789958eb247b5abdaf028aa170fbd5506e188a4c645b48e0ddf9eebf06424)

## 三、数据集
根据复现要求我们用的是 [Cifar10](https://aistudio.baidu.com/aistudio/datasetdetail/103297) 数据集。
* 数据集大小：10类别，训练集有50000张图片。测试集有10000张图片，图像大小为32x32，彩色图像；
* 数据格式：用paddle.vision.datasets.Cifar10调用，格式为cifar-10-python.tar.gz

## 四、环境依赖
* 硬件：使用了百度AI Studio平台的至尊GPU
* 框架：PaddlePaddle >= 2.0.0，平台提供了所有依赖，不必额外下载

## 五、快速开始
cofig.py中提供了论文中提到的默认配置，故以下只按默认配置指导如何使用，如需修改参数可以直接在config.py中修改，或按argparse的用法显式地修改相应参数。


```python
%cd DNSD/
!bash run.sh
```


```python
# 执行main.py，使用默认参数进行模型训练
%cd DNSD/
!python main.py --save_dir checkpoint #--use_official_implement False
```


```python
# 执行main.py，使用默认参数和高层API进行模型训练
%cd DNSD/
!python main.py --epochs 1 --high_level_api True
```


```python
# 执行main.py，使用高层API进行模型评估
%cd DNSD/
!python main.py --high_level_api True --mode eval --checkpoint output/model_best.pdparams
```


```python
# 执行main.py，使用基础API进行模型评估
%cd DNSD/
!python main.py --mode eval --checkpoint output/model_best.pdparams #--use_official_implement False
```


```python
# 执行train.py，使用默认参数进行模型训练
!python train.py # --high_level_api True
```


```python
# 执行eval.py，进行模型评估
%cd DNSD/
!python eval.py --checkpoint output/model_best.pdparams # --high_level_api True
```

## 六、代码结构与详细说明
  几乎完全参考https://github.com/PaddlePaddle/Contrib/wiki/Contrib-%E4%BB%A3%E7%A0%81%E6%8F%90%E4%BA%A4%E8%A7%84%E8%8C%83
  参数详解见config.py每个参数的help信息。

## 七、 参考
  并没有按照比赛提供的参考实现 [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) 中的DropPath来写，因为对比了论文描述和作者实现，发现并不相符。
  * 官方实现 https://github.com/yueatsprograms/Stochastic_Depth
  * 与论文描述比较相符的非官方实现 https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet
  * 论文 https://arxiv.org/pdf/1603.09382v3.pdf
  * 作者推荐的pytorch实现https://github.com/felixgwu/img_classification_pk_pytorch （对比了与我们的很相似）

[](http://)
