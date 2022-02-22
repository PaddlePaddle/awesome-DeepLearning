# 论文复现：Prototypical Networks for Few-shot Learning

## 一、简介

Prototypical networks for few-shot learning是小样本学习方向的一篇经典论文，是一种基于元学习的小样本分类方法。
ProtoNet虽然也是通过构建很多组不同的episode（即元任务）进行训练，但与其他元学习方法的不同之处在于，
ProtoNet目的在于获得一个encoder，将输入图片映射到一个高维的特征空间。在该特征空间中，
support set中每类样本的均值向量，即为该类别的prototype向量，
需要对query样本与各类别的prototype向量求欧式距离，并以此距离作为后续对类别归属及loss函数的构建依据。
完整的算法流程如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/82bb4621e8994b6f96a74825ad363da9fa891178f2a34bfa8d258a66f08a0ee5)

ProtoNet算法的优势在于，其Prototype向量可以通过support样本求均值获得，从而解决小样本分类问题；
还可以通过直接设定类别的高层语义描述，用于零样本分类问题的解决。

![](https://ai-studio-static-online.cdn.bcebos.com/189fe235baa2486b8e3395423a50d29bd7b7d5bd75904972984eb089231106d4)

论文链接：[Prototypical Networks for Few-shot Learning](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)

## 二、复现精度

基于paddlepaddle深度学习框架，对文献算法进行复现后，本项目达到的测试精度，如下表所示。

|task|本项目精度|参考文献精度|
|----|----|----|
|5-way-1-shot|50.16+-0.80|49.42+-0.78|
|5-way-5-shot|68.3+-0.65|68.20+-0.66|


5-way-1-shot超参数配置如下：

|超参数名|设置值|
|----|----|
|data.way|30|
|data.shot|1|
|data.query|15|
|data.test_way|5|
|data.test_shot|1|
|data.test_query|15|
|lr|0.001|

5-way-5-shot超参数配置如下：

|超参数名|设置值|
|----|----|
|data.way|20|
|data.shot|5|
|data.query|15|
|data.test_way|5|
|data.test_shot|5|
|data.test_query|15|
|lr|0.001|


## 三、数据集
miniImageNet数据集节选自ImageNet数据集。
DeepMind团队首次将miniImageNet数据集用于小样本学习研究，从此miniImageNet成为了元学习和小样本领域的基准数据集。
关于该数据集的介绍可以参考https://blog.csdn.net/wangkaidehao/article/details/105531837

miniImageNet是由Oriol Vinyals等在[Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)
中首次提出的，该文献是小样本分类任务的开山制作，也是本次复现论文关于该数据集的参考文献。在Matching Networks中，
作者提出对ImageNet中的类别和样本进行抽取（参见其Appendix B），形成了一个数据子集，将其命名为miniImageNet。
划分方法，作者仅给出了一个文本文件进行说明。
Vinyals在文中指明了miniImageNet图片尺寸为84x84。因此，后续小样本领域的研究者，均是基于原始图像，在代码中进行预处理，
将图像缩放到84x84的规格。

至于如何缩放到84x84，本领域研究者各有各的方法，通常与研究者的个人理解相关，但一般对实验结果影响不大。本次文献论文原文，未能给出
miniImageNet的具体实现方法，本项目即参考领域内较为通用的预处理方法进行处理。

- 数据集大小：
  - miniImageNet包含100类共60000张彩色图片，其中每类有600个样本。
    mini-imagenet一共有2.86GB
- 数据格式：
```
|- miniImagenet
|  |- images/
|  |  |- n0153282900000005.jpg
|  |  |- n0153282900000006.jpg
|  |  |- …
|  |- train.csv
|  |- test.csv
|  |- val.csv
```


数据集链接：[miniImagenet](https://aistudio.baidu.com/aistudio/datasetdetail/105646/0)


## 四、环境依赖

- 硬件：
    - x86 cpu
    - NVIDIA GPU
- 框架：
    - PaddlePaddle = 2.1.2

- 其他依赖项：
    - numpy==1.19.3
    - tqdm==4.59.0
    - Pillow==8.3.1


## 五、快速开始

### 1、解压数据集和源代码：
`!unzip -n -d ./data/ ./data/data105646/mini-imagenet-sxc.zip`


```python
%cd /home/aistudio/
!unzip -n -d ./data/ ./data/data105646/mini-imagenet-sxc.zip
```


```python
%cd /home/aistudio/
!unzip -o prototypical-networks-paddle.zip
```

### 2、执行以下命令启动训练：

```
python run_train.py --data.way 30 --data.shot 1 --data.query 15 --data.test_way 5 --data.test_shot 1 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```

模型开始训练，运行完毕后，有三个文件保存在./results目录下，分别是:

```
best_model.pdparams  # 最优模型参数文件
opt.json  # 训练配置信息
trace.txt  # 训练LOG信息
```

训练完成后，可将上面三个文件手动保存到其他目录下，避免被后续训练操作覆盖。



```python
# 5-way-1-shot训练
%cd /home/aistudio/prototypical-networks-paddle/
!python run_train.py --data.way 30 --data.shot 1 --data.query 15 --data.test_way 5 --data.test_shot 1 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```


```python
# 5-way-5-shot训练
# %cd /home/aistudio/prototypical-networks-paddle/
# !python run_train.py --data.way 20 --data.shot 5 --data.query 15 --data.test_way 5 --data.test_shot 5 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```

### 3、执行以下命令进行评估

```
python run_eval.py --model.model_path results/5w1s/best_model.pdparams --data.test_way 5 --data.test_shot 1 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```

用于评估模型在小样本任务下的精度。



```python
# 5-way-1-shot评估
%cd /home/aistudio/prototypical-networks-paddle/
!python run_eval.py --model.model_path results/5w1s/best_model.pdparams --data.test_way 5 --data.test_shot 1 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```


```python
# 5-way-5-shot评估
%cd /home/aistudio/prototypical-networks-paddle/
!python run_eval.py --model.model_path results/5w5s/best_model.pdparams --data.test_way 5 --data.test_shot 5 --data.test_query 15 --data_root /home/aistudio/data/mini-imagenet-sxc
```

## 六、代码结构与详细说明
### 6.1 代码结构
```
├── images                          # README.md图片
├── protonets                       # 模型代码
│   ├── data                        # 数据相关
│   ├── models                      # 模型相关
│   ├── utils                       # 公共调用
├── README.md                       # readme
├── requirements.txt                # 依赖
├── run_eval.py                     # 执行评估
├── run_train.py                    # 启动训练入口
├── results                         # 结果
│   ├── 5w1s                        # 5w1s最优结果
│   ├── 5w5s                        # 5w5s最优结果
├── scripts                         # 训练和推理脚本
│   ├── predict                     # 推理脚本目录
│   ├── train                       # 训练脚本目录
│   ├── averagevaluemeter.py        # 均值计算
```


### 6.2 参数说明
可以在 `run_train.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| --data.way | 30 | 训练ways | 5-way-1-shot训练时设置为30，5-way-5-shot训练时设置为20 |
| --data.shot | 1 | 训练shots ||
| --data.query | 15 | 训练queries ||
| --data.test_way | 5 | 测试ways ||
| --data.test_shot | 1 | 测试shots ||
| --data.test_query | 15 | 测试queries ||
| --data_root | 必选 | 设置miniImageNet数据集路径 |在本项目所公开的aistudio项目下，路径可配置为/home/aistudio/data/mini-imagenet-sxc|

可以在 `run_eval.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| --model.model_path | results/best_model.pdparams | 要评估的模型参数路径 | 5-way-1-shot的路径为./results/5w1s，5-way-5-shot的路径为./results/5w5s |
| --data.test_way | 5 | 测试ways ||
| --data.test_shot | 1 | 测试shots ||
| --data.test_query | 15 | 测试queries ||
| --data_root | 必选 | 设置miniImageNet数据集路径 |在本项目所公开的aistudio项目下，路径可配置为/home/aistudio/data/mini-imagenet-sxc|



### 6.3 训练流程
可参考快速开始章节中的描述
#### 训练输出
执行训练开始后，将得到类似如下的输出。每一轮`epoch`训练将会打印当前training loss、training acc、val loss、val acc以及训练kl散度。
```text
/home/aistudio/prototypical-networks-paddle
W0830 09:31:39.461673  1253 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
W0830 09:31:39.465984  1253 device_context.cc:422] device: 0, cuDNN Version: 7.6.
Epoch 1 train:   0%|                                    | 0/100 [00:00<?, ?it/s]/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  if data.dtype == np.object:
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/norm.py:641: UserWarning: When training, we now always track global mean and variance.
  "When training, we now always track global mean and variance.")
Epoch 1 train: 100%|██████████████████████████| 100/100 [03:32<00:00,  2.12s/it]
Epoch 1 valid: 100%|██████████████████████████| 100/100 [00:54<00:00,  1.83it/s]
Epoch 01: train loss = 2.733919, train acc = 0.194633, val loss = 1.454196, val acc = 0.404667
==> best model (loss = 1.454196), saving model...
Epoch 2 train: 100%|██████████████████████████| 100/100 [03:32<00:00,  2.13s/it]
Epoch 2 valid: 100%|██████████████████████████| 100/100 [00:53<00:00,  1.86it/s]
Epoch 02: train loss = 2.577872, train acc = 0.232200, val loss = 1.395858, val acc = 0.439867
==> best model (loss = 1.395858), saving model...
Epoch 3 train: 100%|██████████████████████████| 100/100 [03:31<00:00,  2.11s/it]
Epoch 3 valid: 100%|██████████████████████████| 100/100 [00:53<00:00,  1.86it/s]
Epoch 03: train loss = 2.460392, train acc = 0.272867, val loss = 1.317674, val acc = 0.462000
==> best model (loss = 1.317674), saving model...
Epoch 4 train: 100%|██████████████████████████| 100/100 [03:32<00:00,  2.12s/it]
Epoch 4 valid: 100%|██████████████████████████| 100/100 [00:54<00:00,  1.82it/s]
Epoch 04: train loss = 2.329943, train acc = 0.303833, val loss = 1.238956, val acc = 0.498267
==> best model (loss = 1.238956), saving model...
Epoch 5 train: 100%|██████████████████████████| 100/100 [03:32<00:00,  2.13s/it]
Epoch 5 valid: 100%|██████████████████████████| 100/100 [00:53<00:00,  1.85it/s]
Epoch 05: train loss = 2.253531, train acc = 0.320500, val loss = 1.210853, val acc = 0.512133
==> best model (loss = 1.210853), saving model...
```


### 6.4 测试流程
可参考快速开始章节中的描述

此时的输出为：
```
(paddle_gpu) F:\FormalDL\比赛\论文复现赛第四期20210810\Prototypical Networks for Few-shot Learning\prototypical-networks-paddle>python run_eval.py --model.model_path results/best_model.pdparams --data.test_way 5 --data.test_shot 5 --da
ta.test_query 15
W0909 08:13:58.964854 20156 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 11.2, Runtime API Version: 11.0
W0909 08:13:58.972832 20156 device_context.cc:422] device: 0, cuDNN Version: 8.1.
Evaluating 5-way, 5-shot with 15 query examples/class over 600 episodes
test: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [03:02<00:00,  3.29it/s]
test loss: 0.804894 +/- 0.014917
test acc: 0.683000 +/- 0.006517
```


## 七、对比实验及复现心得
### 7.1 实验数据比较

在不同的超参数配置下，模型的收敛效果、达到的精度指标有较大的差异。另外，对数据做不同的预处理，最后达到的实验精度也会有较大影响。
以下分别做论述。

（1）学习率：

原文献采用的优化器与本项目一致，为Adam优化器，原文献学习率设置为0.001；
对于lr_scheduler，原文使用的是StepLR，gamma=0.5，即每间隔20个epoch学习率减半。
实验发现，在训练的前期，20个epoch训练还没有收敛，此时降低学习率不利于模型收敛。
而到了训练后期，20个epoch才调整学习率，又显得时间太长。因此，本项目改用ReduceOnPlateau，
对val_loss进行监测，当10个epoch val_loss不下降时，学习率降低到原来的0.3倍。

（2）epoch轮次

本项目训练时，采用的epoch轮次为400，当100个epoch都没有使val_loss进一步下降的话，训练终止。
LOSS和准确率在10个epoch附近已趋于稳定，模型处于收敛状态，下图为5-way-1-shot的训练曲线。

![](https://ai-studio-static-online.cdn.bcebos.com/3bb98ad6879b4cdcb5c1e491890f9616776b45cdd3d24f68b17860dcbe40b7af)


（3）关于数据扩增

原文献中没有对miniImageNet做数据增强的描述，我们这里应默认为未做数据增强。但出于原理性方面的探究，本项目对是否采用数据增强，增加了对比实验。
实验结果如下：

|task|无扩增|扩增|
|----|----|----|
|5-way-1-shot|50.16+-0.80|49.14+-0.62|
|5-way-5-shot|68.3+-0.65|67.17+-0.66|

分析：小样本分类任务在任务训练方面，有基于元学习的方法和基于迁移学习的方法两种，本文属于前者，即在训练阶段构造一系列小样本任务进行训练。
这就使得，在小样本条件下，数据扩增带来的影响并不是增加了训练数据（shot数没有变），而是增加了额外噪声。这种噪声会劣化模型本来的收敛方向，
导致精度有一定程度降低。因此，在进行基于元学习方法进行小样本任务训练时，应特别注意数据扩增引起的本质变化。

（4）关于数据预处理

由于miniImageNet是ImageNet抽取出来的图像子集，图像分辨率仍然为ImageNet中的原始大小，需要文献按要求预处理为84x84大小的分辨率。
直接进行resize是一种方法，但不合理，这会使得数据集中不同长宽比的图像统一缩放到1:1的比例，引起图像内容的形变。
正确的预处理方法应该是，先对原始图像按短边等比例resize到84，然后在CenterCrop为84x84。这样能有效防止形变发生。

对上述提到的两种预处理方法，根据其是否引起了形变，本项目做了对比实验，实验结果如下表所示：

|task|无形变|有形变|
|----|----|----|
|5-way-1-shot|50.16+-0.80|50.47+-0.83|
|5-way-5-shot|68.3+-0.65|67.98+-0.62|


### 7.2 复现心得
本项目复现时遇到一个比较大的问题，是前期训练完成后精度总是低于文献所述的精度2个百分点。

原文的repo只提供了omniglot数据集的训练代码，miniImageNet的数据读取部分是本项目按小样本方向的典型处理方式编写的，
即进行数据扩增。然而，实验发现ProtoNet在不进行数据增广的情况下，其精度反而更高。我自己的复现实验，也验证了这一点。
我对该现象的解释已在上一小节中阐述。原文献未提及使用数据扩增，也应是相同道理。

ProtoNet文献中指出的另一个有意思的结论是，在该算法下，用欧氏距离度量得到的精度比余弦相似度更高；
这一结论与其他算法得到的结论恰好相反，后续研究值得进一步关注。


## 八、模型信息

训练完成后，模型和相关LOG保存在./results/5w1s和./results/5w5s目录下。

训练和测试日志保存在results目录下。

| 信息 | 说明 |
| --- | --- |
| 发布者 | hrdwsong |
| 时间 | 2021.08 |
| 框架版本 | Paddle 2.1.2 |
| 应用场景 | 小样本学习 |
| 支持硬件 | GPU、CPU |
|Aistudio地址|https://github.com/hrdwsong/ProtoNet-Paddle|


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
