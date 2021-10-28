# 论文简介
<center><img src="https://gitee.com/atari/STDC-Seg/raw/master/images/overview-of-our-method.png"， height=60%, width=60%></center>

## 总体结构 (pipeline)
<center><img src="https://gitee.com/atari/STDC-Seg/raw/master/images/stdcseg-architecture.png"， height=60%, width=60%></center>

网络的特征提取部分是文章设计的模型构成，名为STDC模型。共两个不同的型号：STDCNet813（下文称为STDCNet1、STDCNet1446（下文称为STDCNet2））。然后借鉴了BiSiNet的结构设计，加入ARM模块和FFM模块。同时为了更好的学习边界细节信息，加入专门的Detail Head。利用辅助分割头提升分割性能，最后实现了高效的分割。

## Backbone & Neck
<center><img src="https://gitee.com/atari/STDC-Seg/raw/master/images/stdc-architecture.png"， height=60%, width=60%></center>

作者重新思考了分割任务与分类任务的不同：分类任务中，深层的通道比浅层多，深层的高级语义信息对分类任务更加重要；分割任务中，需要同时关注深层和浅层信息。浅层（小感受野）需要足够多的通道来提取更多精细的信息，高层（大感受野）通道如果和浅层一样多的话，会造成冗余，所以高层通道越来越少。由此设计了STDC（Short-Term Dense Concatenate Module ）模块：其特点是只有第一个block的卷积核大小为1x1，其余都为3x3，浅层通道多，深层通道少。并且只有 Block2 进行了下采样，利用跳跃连接实现多级特征的局部融合。通过这种设计，显著降低了计算复杂度，保留了多尺度感受野和多尺度信息。

**两种不同的STDC Network**：

<center><img src="https://img-blog.csdnimg.cn/2021042920240860.png
"， height=60%, width=60%></center>


## Seghead
如pipeline图所示，作者使用stage 3、4、5来生成下采样比率分别为 1/8、1/16、1/32的特征图。然后使用全局平均池化来得到语义信息。之后，使用U-shape结构来对全局特征进行上采样，并且和stage4、stage5的进行结合（在encoder阶段）。

**context info 和 spatial info 的结合使用**：

在最后的语义分割预测阶段，作者使用了 Feature Fusion Module，来融合来自encoder 的 stage3 (1/8) 和 decoder 的stage3的特征。encoding 的特征有更多的细节信息，decoding的特征有更多的语义信息（自于global average pooling）。
Seg Head 的构成：一个 3x3 conv+bn+relu，再跟一个 1x1 卷积，输出维度为类别数量。

## Detail Head

<center><img src="https://img-blog.csdnimg.cn/20210430103322963.png
"， height=20%, width=20%></center>

为了更好的指导边界细节信息的学习，文章提出了一个 Detail Guidance Module。将细节预测建模为一个二值分割任务，利用不同尺度的的Laplacian核从segmentation gt生成不同尺度的detail gt，融合输出detailed gt，通过二值化得到最终的二值gt图，而网络的细节输出则从stage3接上一个Detail Head生成。通过这一操作，实现了边缘细节信息的学习。

## Loss

损失主要包含两个方面：一个是语义分割头的损失计算;另外则是detail head的损失计算。
语义分割头的损失实际采用的是：OhemCrossEntropyLoss;
而细节损失由于边界像素相较于非边界像素的数量是很不平衡的，因为为了解决这一问题，文章采用了binary crossentropy & dice loss联合学习。

## 实验

### 训练
原论文优化器选择了 Momentum，power=0.9，batch-size 为 48，4 块 V100 GPU。学习率采用了warmup策略，先在1000iter上升到0.01，之后采用polydecay的策略。总共iters为60000。作为Backbone的STDCNet的预训练权重是在 ImageNet 上训练的。采用了0.125到1.5（步长0.125）的多尺度训练。并且对数据进行了随机翻转等增强操作。训练的图像crop到1024*512。

同时，论文进行了大量消融实验，通过实验验证了在 STDC 中使用4个 block 是最优的。并且细节部分的gt采用多尺度融合的结果最好。
<center><img src="https://img-blog.csdnimg.cn/20210430111857642.png"， height=50%, width=50%></center>

更加详细的内容可以参考论文原文.

# 项目介绍

## 项目背景
项目为第四届百度论文复现赛Rethinking BiSiNet（CVPR2021）第一名比赛结果。项目基于 <font color=red>Paddle 2.0.2</font> 与 <font color=red>paddleseg</font> 进行开发并实现论文精度，十分感谢百度提供比赛平台和 GPU 资源！

## 项目结果

Method|Environment|mAP|iters|Dataset
:--:|:--:|:--:|:--:|:--:
STDC2-Seg50|Pytorch with Tesla V-100 $\times$ 4 (Paper)|74.2|60000|Cityscapes
STDC2-Seg50|Paddle with Tesla V-100 $\times$ 1（本项目）|<font color=red>74.62</font>|80000|Cityscapes

&clubs; <font face=Times new roman> 由于只使用了一张V100，所以相较于原文采用了更小的batch_size=36，原文batch_size=48，以及更长的迭代次数80000。其余设置一致。

## 项目实现思路
首先我们在阅读完论文之后，首先得知道算法的结构是怎么样的。然后对原文提供的源码进行剖析，逐一代码对齐。最后需要输入同样的数据，进行输出的对齐。然后判断是否可以使用paddle已经提供的优质库进行快速开发。对于本项目而言，在完成网络结构的对齐和损失函数的编写后。可以很方便的使用paddleseg提供的数据集加载处理、模型训练评估工具进行快速开发。下面是项目的基本实现顺序：1、首先是数据集的准备;2、然后是模型的定义和预训练模型的加载;3、训练集和验证集的加载定义;4、进行训练

**注意：本项目提供两种训练方式：一是在notebook直接按cell运行;二是在notebook处理好数据集以后在终端命令行用python train.py的指令进行训练。他们的差别在于多进程读取数据的采用与否（notebook由于管理限制暂不支持）。**

# 数据集和环境准备(直接每个cell运行)

## cityscapes数据集初始化


```python
# 准备数据集
# 查看当前工作目录
!pwd
# 查看工作区文件夹
!tree -d work/
# 查看数据文件夹
!tree -d data/
```


```python
# 创建cityscape文件夹
!mkdir data/cityscapes/
```


```python
# 解压数据集中附带的官方处理代码
# !unzip -nq -d work/ data/data48855/cityscapesscripts.zip
# !cd work/cityscapesscripts/
```


```python
# 解压数据集中的gtFine
!unzip -nq -d data/gtFine/ data/data48855/gtFine_train.zip
!unzip -nq -d data/gtFine/ data/data48855/gtFine_val.zip
!unzip -nq -d data/gtFine/ data/data48855/gtFine_test.zip
!mv data/gtFine/ data/cityscapes/
```


```python
# 解压数据集中的leftImg8bit
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_train.zip
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_val.zip
!unzip -nq -d data/leftImg8bit/ data/data48855/leftImg8bit_test.zip
!mv data/leftImg8bit/ data/cityscapes/
```


```python
# 查看工作区文件夹
# !tree -d work/
# 查看数据文件夹
!tree  -d data/cityscapes/
```


```python
# 下载PaddleSeg库
!git clone https://gitee.com/paddlepaddle/PaddleSeg.git

```


```python
# 安装paddleseg
!pip install paddleseg
```


```python
# 生成cityscapes文件列表，其分隔符为逗号
!python  PaddleSeg/tools/create_dataset_list.py /home/aistudio/data/cityscapes/ --type cityscapes --separator ","
```

# 应有的数据目录格式
```
data/
├── cityscapes
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── leftImg8bit
│   │   ├── test
│   │   │   ├── berlin
│   │   │   ├── ...
│   │   │   └── munich
│   │   ├── train
│   │   │   ├── aachen
│   │   │   ├── ...
│   │   │   └── zurich
│   │   └── val
│   │       ├── frankfurt
│   │       ├── lindau
│   │       └── munster
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
```

## 模型训练
### 数据准备好以后：参考[https://paddleseg.readthedocs.io/zh_CN/release-2.1/design/create/add_new_model.html](https://paddleseg.readthedocs.io/zh_CN/release-2.1/design/create/add_new_model.html)
几点注意：
1、修改数据集的yml文件路径：指向/home/aistudio/data目录下
2、其他的按文档：比如训练：在PaddleSeg目录下python train.py --config config文件.....

### 1. 构建模型
#### todo :  参考[https://paddleseg.readthedocs.io/zh_CN/release-2.1/design/create/add_new_model.html](https://paddleseg.readthedocs.io/zh_CN/release-2.1/design/create/add_new_model.html)定义自己的model


```python
# 参数设定
# 参数设定
import paddle
from models.model_stages_paddle import BiSeNet as STDCNet
import paddleseg.transforms as T
from paddleseg.datasets import Cityscapes
from paddleseg.models.losses import OhemCrossEntropyLoss
from loss.detail_loss_paddle import DetailAggregateLoss
from tool.train import train
from paddleseg.core import evaluate
from scheduler.warmup_poly_paddle import Warmup_PolyDecay
backbone = 'STDCNet1446' # STDC2: STDCNet1446 ; STDC1: STDCNet813
n_classes = 19
pretrain_path = 'pretrained/STDCNet1446_76.47.pdiparams' # backbone预训练模型参数
use_boundary_16 = False
use_boundary_8 = True
use_boundary_4 = False
use_boundary_2 = False
use_conv_last = False
```


```python
# 模型导入
# 模型导入
model = STDCNet(backbone=backbone, n_classes=n_classes, pretrain_model=pretrain_path,
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
    use_boundary_16=use_boundary_16, use_conv_last=use_conv_last)

# in_ten = paddle.randn((1, 3, 768, 1536))
# print(len(net(in_ten)))

```

### 2. 构建训练集


```python
# 构建训练用的transforms
transforms = [  
    T.ResizeStepScaling(min_scale_factor=0.125,max_scale_factor=1.5,scale_step_size=0.125),  
    T.RandomHorizontalFlip(),
    T.RandomPaddingCrop(crop_size=[1024,512]), #Seg50:imgsize=(512,1024); Seg75:imgsize=(768,1536)
    T.RandomDistort(brightness_range=0.5,contrast_range=0.5,saturation_range=0.5),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

# 构建训练集
train_dataset = Cityscapes(
    dataset_root='/home/aistudio/data/cityscapes', # 数据集路径
    transforms=transforms,
    mode='train'
)


```

### 3. 构建验证集


```python
# 构建验证用的transforms
transforms_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

# 构建验证集
val_dataset = Cityscapes(
    dataset_root='/home/aistudio/data/cityscapes',
    transforms=transforms_val,
    mode='val'
)
```

### 4. 构建优化器


```python
# 设置学习率
base_lr = 0.01
# lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=60000,end_lr=0.00001)
lr = Warmup_PolyDecay(lr_rate=base_lr,warmup_steps=1000,iters=80000,end_lr=1e-5)
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=5.0e-4)
```

### 5. 构建损失函数
为了适应多路损失，损失函数应构建成包含'types'和'coef'的dict，如下所示。  其中losses['type']表示损失函数类型， losses['coef']为对应的系数。需注意len(losses['types'])应等于len(losses['coef'])。


```python
losses = {}
losses['types'] = [OhemCrossEntropyLoss(),OhemCrossEntropyLoss(),OhemCrossEntropyLoss(),DetailAggregateLoss()]
losses['coef'] = [1]*4
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:143: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:


### 6.训练


```python
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    val_scales = 0.5, # miou50
    aug_eval = True,
    optimizer=optimizer,
    # resume_model='output/iter_36400',
    save_dir='output',
    iters=80000,
    batch_size=36,
    save_interval=200,
    log_iters=10,
    num_workers=0, # 非多进程
    losses=losses,
    use_vdl=True)
```

### 7.评估



```python
# 参数设定
import paddle
from models.model_stages_paddle import BiSeNet as STDCSeg
import paddleseg.transforms as T
from paddleseg.datasets import Cityscapes
from paddleseg.models.losses import OhemCrossEntropyLoss
from loss.detail_loss_paddle import DetailAggregateLoss
from paddleseg.core import evaluate
from scheduler.warmup_poly_paddle import Warmup_PolyDecay

backbone = 'STDCNet1446' # STDC2: STDCNet1446 ; STDC1: STDCNet813
n_classes = 19 # 数据类别数目
pretrain_path = None # backbone预训练模型参数
use_boundary_16 = False
use_boundary_8 = True # 论文只用了use_boundary_8，效果最好
use_boundary_4 = False
use_boundary_2 = False
use_conv_last = False

# 模型导入
model = STDCSeg(backbone=backbone, n_classes=n_classes, pretrain_model=pretrain_path,
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
    use_boundary_16=use_boundary_16, use_conv_last=use_conv_last)

#加载模型参数训练（如果没有预训练参数就把下面两行注释掉）
params_state = paddle.load(path='output/best_model/model.pdparams')
model.set_dict(params_state)


# 构建训练用的transforms
transforms = [
    T.ResizeStepScaling(min_scale_factor=0.125,max_scale_factor=1.5,scale_step_size=0.125),
    T.RandomHorizontalFlip(),
    T.RandomPaddingCrop(crop_size=[1024,512]), #Seg50:imgsize=(512,1024); Seg75:imgsize=(768,1536)
    T.RandomDistort(brightness_range=0.5,contrast_range=0.5,saturation_range=0.5),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

# 构建训练集
train_dataset = Cityscapes(
    dataset_root='data/cityscapes', # 修改数据集路径
    transforms=transforms,
    mode='train'
)

# 构建验证用的transforms
transforms_val = [
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]

# 构建验证集
val_dataset = Cityscapes(
    dataset_root='data/cityscapes',
    transforms=transforms_val,
    mode='val'
)


# 设置学习率
base_lr = 0.01
# lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=60000,end_lr=0.00001)
lr = Warmup_PolyDecay(lr_rate=base_lr,warmup_steps=1000,iters=80000,end_lr=1e-5)
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=5.0e-4)

losses = {}
losses['types'] = [OhemCrossEntropyLoss(),OhemCrossEntropyLoss(),OhemCrossEntropyLoss(),DetailAggregateLoss()]
losses['coef'] = [1]*4

evaluate(model,
        val_dataset,
        aug_eval=True,
        scales=0.5,  # m50; m75: scales=0.75
        flip_horizontal=False,
        flip_vertical=False,
        is_slide=False,
        stride=None,
        crop_size=None,
        num_workers=0,
        print_detail=True)
```

# 说明（重要）

若不想像上述过程操作，可以打开终端运行 python train.py实现多进程读取数据训练，提高训练速度。同时，可以很方便的将该模型集成道paddleseg库中（TODO）。另外，对于不同的模型设置，可以参见论文进行backbone的修改和评估的设置。另外，本项目output/best_model/文件夹下存放有复现的STDC2-Seg50的模型训练好的参数miou=74.62。可直接load以后在上面加载数据集以后利用paddleseg的评估工具进行评估。github上对本项目有详细说明。

## 目前已经集成道Paddleseg，可在PaddleSeg官方仓库找到试用。


# 关于论文

--> [<font face=宋体 size="3">论文链接</font>](https://arxiv.org/abs/2104.13188) </br>
--> [<font face=宋体 size="3">论文详解</font>](https://blog.csdn.net/jiaoyangwm/article/details/116272944) </br>
--> [<font face=宋体 size="3">论文 github 地址</font>](https://github.com/MichaelFan01/STDC-Seg) </br>
--> [<font face=宋体 size="3">复现 github 地址</font>](https://github.com/CuberrChen/STDCNet-Paddle/tree/master)
