# 对抗性自编码器：基于飞桨复现AAE

# Adversarial Autoencoders

<!-- vscode-markdown-toc -->
* 1. [1、简介](#)
* 2. [2、复现精度](#-1)
    * 2.1. [2.1 生成图片](#-1)
    * 2.2. [2.2 loss](#loss)
    * 2.3. [2.3 Likelihood](#Likelihood)
        * 2.3.1. [2.3.1 Parzen 窗长σ在验证集上交叉验证曲线](#Parzen)
        * 2.3.2. [2.3.2 复现结果](#-1)
        * 2.3.3. [2.3.3 对比实验](#-1)
* 3. [3、数据集](#-1)
* 4. [4、环境依赖](#-1)
* 5. [5、快速开始](#-1)
* 6. [6、代码结构与详细说明](#-1)
    * 6.1. [6.1 代码结构](#-1)
    * 6.2. [6.2 参数说明](#-1)
    * 6.3. [6.3 实现细节说明](#-1)
    * 6.4. [6.4 训练流程](#-1)
    * 6.5. [6.5 测试流程](#-1)
    * 6.6. [6.6 使用预训练模型评估](#-1)
* 7. [7、模型信息](#-1)

<!-- vscode-markdown-toc-config
    numbering=true
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


##  1. <a name=''></a>1、简介

本项目基于paddlepaddle框架复现Adversarial Autoencoders (AAE)。这是一种概率自编码器，它使用生成对抗网络(GAN)，通过匹配编码器（Encoder）隐变量的聚合后验概率分布与任意先验分布来执行变分推理。 它将聚合的后验与先验分布进行匹配，确保从任何先验空间中的部分生成有意义的样本。 AAE的译码器（Decoder）学习一个深度生成模型，该模型再映射所给定的先验分布到对应的数据上。

**论文:**

> [1] Makhzani A, Shlens J, Jaitly N, et al. Adversarial autoencoders[J]. arXiv preprint arXiv:1511.05644, 2015.

**参考项目：**

1. https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/aae/aae.py
2. https://github.com/fducau/AAE_pytorch/tree/master/

**项目aistudio地址：**

- notebook任务：https://aistudio.baidu.com/aistudio/projectdetail/2301660

##  2. <a name='-1'></a>2、复现精度

###  2.1. <a name='-1'></a>2.1 生成图片
| Epoch0                                                       | Epoch20                                                      | Epoch100                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://ai-studio-static-online.cdn.bcebos.com/967972ee668b402b8ed7d2bc9dbcf3de53a58721e63541f697786448c4c6a71d)|  ![](https://ai-studio-static-online.cdn.bcebos.com/e264f6fd61cc41f2acaa1ca5c33de5208dc9884055374537b14d7505cf33e263)| ![](https://ai-studio-static-online.cdn.bcebos.com/66ce25e920a543bf84d44f597bceb5243a74b6979c4e45f3a4480cd86bfd9ff0)|


###  2.2. <a name='loss'></a>2.2 loss
<img src="https://ai-studio-static-online.cdn.bcebos.com/6b025dc85477444893d0a52224a27992858e9ab3d0614889844b5ba3445e7db4" width = "750" align=center />

图中，D_loss 为判别器的loss，G_loss 为编码器的loss，recon_loss 为图片的loss，此处选二元交叉熵函数（BCEloss）

recon loss = BCE(X_{sample}, X_{decoder})

如图所示，D_loss 与G_loss在epoch为50轮左右达到稳定。而recon_loss则逐渐减小，在20-30轮稳定。

###  2.3. <a name='Likelihood'></a>2.3 Likelihood

####  2.3.1. <a name='Parzen'></a>2.3.1 Parzen 窗长σ在验证集上交叉验证曲线

原文中提到对于窗长σ的选取需要通过交叉验证实现。本文使用1000个数据样本的验证集进行选择，从而得到一个符合最大似然准则下的窗长。

<img src="https://ai-studio-static-online.cdn.bcebos.com/fc1cc33731c64ee2a5a385001f85d9e52bae016c769d4d72aa53f90919985216" width = "600" align=center />

根据少量样本验证集的结果，选择使得negative log likelihood（nll）最大的σ值作为窗长

####  2.3.2. <a name='-1'></a>2.3.2 复现结果
|      | MNIST (10K)                                                  |
| ---- | ------------------------------------------------------------ |
| 原文 | 340 ± 2                                                   |
| 复现 | 345 ± 1.9432                                              |
|      | ![](https://ai-studio-static-online.cdn.bcebos.com/297396923f824dedb8a9bf7cd78532dd267bc8fa4f364d6893c1e61138344c0c) |

####  2.3.3. <a name='-1'></a>2.3.3 对比实验

|                       | MNIST (10K)  |
| --------------------- | ------------ |
| 原文                  | 340 ± 2      |
| 当前模型              | 345 ± 1.9432 |
| 参考项目[1]中模型     | 298 ± 1.7123 |
| 当前模型去除dropout层 | 232 ± 2.0113 |

**实验结果分析：** 结果表明，对于当前模型设置能够最接近甚至超过原文的nll指标。该指标一方面衡量了生产数据样本的多样性，另一方面又要求生成的数据样本尽可能接近数据集中的样本。去除dropout层的模型容易产生过拟合，模型倾向于保守的策略进行样本生成，数据多样性较差。参考项目[1]中模型使用了LeakyReLU激活函数，效果有所提升，但同样因没有dropout层而容易产生过拟合。

##  3. <a name='-1'></a>3、数据集

[MNIST](http://yann.lecun.com/exdb/mnist/)

- 数据集大小：
  - 训练集：60000
  - 测试集：10000
- 数据格式：idx格式

##  4. <a name='-1'></a>4、环境依赖

- 硬件：GPU、CPU
- 框架：
  - PaddlePaddle >= 2.0.0
  - numpy >= 1.20.3
  - matplotlib >= 3.4.2
  - pandas >=  1.3.1

##  5. <a name='-1'></a>5、快速开始

- **step1:** 数据生成

```
python data_maker.py
```

- **step2:** 训练

```
python train.py
```

- **step3:** 测试log_likelihood

```
python eval.py
```

##  6. <a name='-1'></a>6、代码结构与详细说明

###  6.1. <a name='-1'></a>6.1 代码结构

```
AAE_paddle_modified
├─ README_cn.md                   # 中文readme
├─ README.md                      # 英文readme
├─ data                           # 存储数据集和分割的数据
├─ images                         # 存储生成的图片
├─ logs                           # 存储实验过程log输出
├─ model                          # 存储训练模型
├─ utils                          # 存储工具类代码
   ├─ log.py                      # 输出log
   ├─ paddle_save_image.py        # 输出图片
   └─ parzen_ll.py                # parzen窗估计
├─ config.py                      # 配置文件
├─ network.py                     # 网络结构
├─ data_maker.py                  # 数据分割
├─ train.py                       # 训练代码
└─ eval.py                        # 评估代码
```

###  6.2. <a name='-1'></a>6.2 参数说明

可以在 `config.py` 中设置训练与评估相关参数，主要分为三类：与模型结构相关、与数据相关和与训练测试环境相关。

###  6.3. <a name='-1'></a>6.3 实现细节说明

- 遵循原文原则，Encoder采用两层全连接层，Decoder采用两层全连接层，Discriminator采用两层全连接层。由于原文并未公布代码及相关详细参数，因此笔者在此处加入了Dropout层，以及按照原文附录中所述加入了re-parametrization trick ，即将生成的隐变量z重新参数化为高斯分布。
- 高斯先验分布为8维，方差为5，神经元数为1200. 网络结构与原文一致

###  6.4. <a name='-1'></a>6.4 训练流程

运行

```
python train.py
```

在终端会产生输出，并会保存到`./logs/train.log`中

```
[2021-09-22 21:04:17,682][train.py][line:62][INFO] Namespace(n_epochs=200, batch_size=100, gen_lr=0.0002, reg_lr=0.0001, load=False, N=1200, img_size=28, channels=1, latent_dim=8, std=5, N_train=59000, N_valid=1000, N_test=10000, N_gen=10000, batchsize=100, load_epoch=199, sigma=None)
W0922 21:04:18.062372 50879 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 10.1, Runtime API Version: 10.1
W0922 21:04:18.067184 50879 device_context.cc:422] device: 0, cuDNN Version: 7.6.
[2021-09-22 21:04:48,115][train.py][line:174][INFO] [Epoch 0/200] [Batch 589/590] [D loss: 1.399173] [G loss: 0.836521] [recon loss: 0.201558]
[2021-09-22 21:04:48,154][train.py][line:181][INFO] images0 saved in ./images/images0.png
[2021-09-22 21:05:15,312][train.py][line:174][INFO] [Epoch 1/200] [Batch 589/590] [D loss: 1.831465] [G loss: 0.627008] [recon loss: 0.163741]
```

###  6.5. <a name='-1'></a>6.5 测试流程

运行

```
python eval.py
```

在终端会产生输出，并会保存到`./logs/eval.log`中

```
[2021-09-22 22:36:11,013][eval.py][line:29][INFO] Namespace(n_epochs=200, batch_size=100, gen_lr=0.0002, reg_lr=0.0001, load=False, N=1200, img_size=28, channels=1, latent_dim=8, std=5, N_train=59000, N_valid=1000, N_test=10000, N_gen=10000, batchsize=100, load_epoch=199, sigma=None)
W0922 22:36:12.574378 66119 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 10.1, Runtime API Version: 10.1
W0922 22:36:12.579428 66119 device_context.cc:422] device: 0, cuDNN Version: 7.6.
[2021-09-22 22:36:14,076][eval.py][line:40][INFO] model/model199.pkl loaded!
[2021-09-22 22:37:04,696][parzen_ll.py][line:32][INFO] sigma = 0.10000, nll = 134.13950
[2021-09-22 22:37:53,652][parzen_ll.py][line:32][INFO] sigma = 0.10885, nll = 214.54500
```

###  6.6. <a name='-1'></a>6.6 使用预训练模型评估

预训练模型保存在[aistudio项目](https://aistudio.baidu.com/aistudio/projectdetail/2301660)中的`./model/model199.pkl`，为第199轮输出结果。可以快速对模型进行评估。如`./data/`文件夹下无分割后的数据集，请先运行`datamaker.py`产生分割后的数据集。


##  7. <a name='-1'></a>7、模型信息

关于模型的其他信息，可以参考下表：

| 信息     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| 发布者   | 钟晨曦                                                       |
| 时间     | 2021.08                                                      |
| 框架版本 | Paddle 2.1.2                                                 |
| 应用场景 | 数据降维                                                     |
| 支持硬件 | GPU、CPU                                                     |
| 下载链接 | [预训练模型](https://aistudio.baidu.com/aistudio/projectdetail/2301660) |
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/2301660) |



## Step 0: 配置参数
设置训练、测试参数



```python
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--gen_lr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--reg_lr", type=float, default=0.0001, help="adam: reconstruction learning rate")
    parser.add_argument("--load", type=bool, default=False, help="load model or not")
    parser.add_argument("--N", type=bool, default=1200, help="Number of neurons")
    # data
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")
    parser.add_argument("--latent_dim", type=int, default=8, help="dimensionality of the latent code")
    parser.add_argument("--std", type=float, default=5, help="std prior")
    parser.add_argument("--N_train", type=int, default=59000, help="Number of training data samples")
    parser.add_argument("--N_valid", type=int, default=1000, help="Number of valid data samples")
    parser.add_argument("--N_test", type=int, default=10000, help="Number of test data samples")
    parser.add_argument("--N_gen", type=int, default=10000, help="Number of generated data samples")
    parser.add_argument('--batchsize', type=int, default=100, help='input batch size for training (default: 100)')
    # test
    parser.add_argument("--load_epoch", type=int, default=199, help="the load model id")
    parser.add_argument('--sigma', type=float, default=None, help = "Window width")

    args = parser.parse_known_args()[0]
    return args


```

## Step1: 导入数据
原文中进行似然性估计的数据集数量为10k,因此进行数据集切分
注:此处由于原文中并未提到输入数据的归一化,为保证数据的合理性和最终结果的准确性,在导入数据时也不做归一化操作.


```python
from __future__ import print_function
import os
import pickle
import numpy as np

import paddle
from paddle.io import random_split
from paddle.io import DataLoader
from paddle.vision import datasets
import paddle.vision.transforms as transforms
from utils.log import get_logger
from config import args_parser

opt = args_parser()
# # Configure data loader
logger = get_logger('./logs/data_maker.log')
logger.info('start create datasets!')

os.makedirs("./model", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./images", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

trainset = datasets.MNIST(
        mode="train", download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),]
        ),
    )
testset = datasets.MNIST(
        mode="test", download=True,
        transform=transforms.Compose(
            [transforms.ToTensor()]
        ),
    )
trainset, validset = random_split(trainset, [opt.N_train, opt.N_valid])

traindataloader = DataLoader(
    trainset, batch_size=opt.batchsize, shuffle=True,
)
validdataloader = DataLoader(
    validset, batch_size=opt.N_valid, shuffle=True,
)
testdataloader = DataLoader(
    testset, batch_size=opt.N_test, shuffle=True,
)

valid_imgs, _ = next(iter(validdataloader))
test_imgs, _ = next(iter(testdataloader))

paddle.save(trainset, "./data/train")
paddle.save(valid_imgs, "./data/valid")
paddle.save(test_imgs, "./data/test")

logger.info('finish create datasets!')
```

    [2021-09-29 16:54:34,346][<ipython-input-2-574c19c099d4>][line:17][INFO] start create datasets!
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz
    Begin to download

    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz
    Begin to download
    ........
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/t10k-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-images-idx3-ubyte.gz
    Begin to download

    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/t10k-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/t10k-labels-idx1-ubyte.gz
    Begin to download
    ..
    Download finished
    [2021-09-29 16:54:44,761][<ipython-input-2-574c19c099d4>][line:55][INFO] finish create datasets!


## Step2: 训练
### AAE结构
![](https://ai-studio-static-online.cdn.bcebos.com/5ac6c8150cbc4bdab78f31d03dd8ecdfbeb3d2272148459fa7595ccd392a31c7)

**由编码器(Encoder),译码器(Decoder),判别器(Discriminator)构成**

## Step2.1: 搭建网络

基于原文附录搭建网络


```python

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.vision import datasets
from paddle.vision.datasets import MNIST
from paddle.io  import random_split
from paddle.io import DataLoader
import pickle
from config import args_parser

opt = args_parser()
N = opt.N
STD = opt.std
z_dim = opt.latent_dim
img_size = opt.img_size

def reparameterization(mu, logvar):
    std = paddle.exp(logvar / 2)
    sampled_z = paddle.normal(0, STD, (mu.shape[0], z_dim))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1*img_size*img_size, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.mu = nn.Linear(N, z_dim)
        self.logvar = nn.Linear(N, z_dim)
        self.direct = nn.Linear(N, z_dim)

    def forward(self, img):
        img_flat = paddle.reshape(img, shape = (img.shape[0], -1) )
        # 编码输出
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Layer):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, 1*img_size*img_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = paddle.reshape(img_flat, shape = [img_flat.shape[0], 1,img_size,img_size] )
        return img


class Discriminator(nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, N),
            nn.Dropout(0.2),
            nn.ReLU( ),
            nn.Linear(N, 1),
            nn.Sigmoid(),
        )  

    def forward(self, z):
        validity = self.model(z)
        return validity
```

### Step2.2: 训练

一共训练两部分模型:

1: 计算重建误差: Encoder-Decoder

2: 计算判别误差 Discriminator (监督编码的效果)


```python
import argparse
import os
import numpy as np
import math
import itertools
import pickle
import pandas as pd

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.io import random_split
from paddle.io import DataLoader
from paddle.vision import datasets
from paddle.vision.datasets import MNIST

from network import Encoder, Decoder, Discriminator
from utils.paddle_save_image import save_image
from utils.parzen_ll import *
from utils.log import get_logger
from config import args_parser

# paddle.utils.run_check()

def sample_image(n_row, epoch):
    """Saves a grid of generated digits"""
    # Sample noise
    z = paddle.normal(0, opt.std, (n_row ** 2, opt.latent_dim))
    gen_imgs = decoder(z)
    # gen_imgs = paddle.to_tensor(gen_imgs)
    save_image(gen_imgs, "images/epoch%3d.png" % epoch, nrow=n_row, normalize=True)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.__class__.weight_attr = nn.initializer.Normal(0.0, 0.01)
    elif classname.find("BatchNorm2d") != -1:
        m.__class__.weight_attr = nn.initializer.Normal(1.0, 0.02)
        m.__class__.weight_attr = nn.initializer.Constant(0.0)

def pd_one_epoch_to_csv(export_data, epoch, D_loss, G_loss, PATH, recon_loss=None):
    export_data_line = np.zeros(3)
    export_data_line[0] = D_loss.item()
    export_data_line[1] = G_loss.item()
    export_data_line[2] = recon_loss.item()
    export_data.append(export_data_line.reshape(-1,))
    data = np.array(export_data)
    data = pd.DataFrame(data=data)
    data.to_csv(PATH,index = True)
    return export_data

if __name__ == "__main__" :
    # Training settings
    opt = args_parser()
    loss = []
    device_id = 0
    os.makedirs("images", exist_ok=True)

    # log 输出
    logger = get_logger('./logs/train.log')
    logger.info(opt)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Configure data loader
    trainset = paddle.load("./data/train")
    traindataloader = DataLoader(
        trainset, batch_size=opt.batchsize, shuffle=True,
    )

    # Initialize generator and discriminator
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator()

    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    # Optimizers
    # Set optimizators
    P_decoder = paddle.optimizer.Adam(
        parameters = decoder.parameters(),
        learning_rate=opt.gen_lr)
    Q_encoder = paddle.optimizer.Adam(
        parameters = encoder.parameters(),
        learning_rate=opt.gen_lr)
    Q_generator = paddle.optimizer.Adam(
        parameters = encoder.parameters(),
        learning_rate=opt.reg_lr)
    D_gauss_solver = paddle.optimizer.Adam(
        parameters = discriminator.parameters(),
        learning_rate=opt.reg_lr)

    if opt.load == True:
        checkpoint = paddle.load("./model/model" + str(opt.load_epoch) + ".pkl")
        encoder.set_state_dict(checkpoint['encoder'])
        decoder.set_state_dict(checkpoint['decoder'])
        discriminator.set_state_dict(checkpoint['discriminator'])
        P_decoder.set_state_dict(checkpoint['P_decoder'])
        Q_encoder.set_state_dict(checkpoint['Q_encoder'])
        Q_generator.set_state_dict(checkpoint['Q_generator'])
        D_gauss_solver.set_state_dict(checkpoint['D_gauss_solver'])

    TINY = 1e-15
    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(traindataloader):
            encoder.train()
            decoder.train()
            discriminator.train()

            # Adversarial ground truths
            valid = paddle.to_tensor(np.ones((imgs.shape[0], 1)), dtype='float32', stop_gradient = True)
            fake = paddle.to_tensor(np.zeros((imgs.shape[0], 1)), dtype='float32', stop_gradient = True)
            # Configure input
            real_imgs = imgs

            #######################
            # Reconstruction phase
            #######################
            z_sample = encoder(real_imgs)
            X_sample = decoder(z_sample)
            recon_loss = F.binary_cross_entropy(X_sample + TINY, real_imgs + TINY)

            recon_loss.backward()
            P_decoder.step()
            Q_encoder.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()
            #######################
            # Regularization phase
            #######################
            # Discriminator
            encoder.eval()
            z_real_gauss = paddle.normal(0, opt.std, (imgs.shape[0], opt.latent_dim))
            z_fake_gauss = encoder(real_imgs)
            z_fake_gauss.stop_gradient = True # 阻止梯度回传
            D_real_gauss = discriminator(z_real_gauss)
            D_fake_gauss = discriminator(z_fake_gauss)

            D_loss = -paddle.mean(paddle.log(D_real_gauss + TINY) + paddle.log(1 - D_fake_gauss + TINY))
            D_loss.backward()
            D_gauss_solver.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()
            # Generator
            encoder.train()
            z_fake_gauss = encoder(real_imgs)
            D_fake_gauss = discriminator(z_fake_gauss)
            G_loss = -paddle.mean(paddle.log(D_fake_gauss + TINY))

            G_loss.backward()
            Q_generator.step()

            P_decoder.clear_grad()
            Q_encoder.clear_grad()
            Q_generator.clear_grad()
            D_gauss_solver.clear_grad()

        encoder.eval()
        decoder.eval()
        discriminator.eval()

        logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [recon loss: %f]"
                % (epoch, opt.n_epochs, i, len(traindataloader), D_loss.item(), G_loss.item(), recon_loss.item())
            )
        loss = pd_one_epoch_to_csv(loss, epoch, D_loss, G_loss, "./logs/loss.csv", recon_loss = recon_loss)
        if epoch % 10 == 0:
            sample_image(n_row=10, epoch=epoch)
            logger.info("images%d saved in ./images/images%d.png" % (epoch, epoch))
        # 计算结果
        if (epoch+1) % 50 == 0:
            logger.info("model%d saved in ./model/model%d.pkl" % (epoch, epoch))
            paddle.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator':discriminator.state_dict(),
                'P_decoder': P_decoder.state_dict(),
                'Q_encoder': Q_encoder.state_dict(),
                'Q_generator': Q_generator.state_dict(),
                'D_gauss_solver': D_gauss_solver.state_dict(),
            },
            str("./model/model" + str(epoch) + ".pkl") )

    logger.info("finish training")





```

    [Epoch 0/200] [Batch 589/590] [D loss: 1.727609] [G loss: 0.682102] [recon loss: 0.193023]
    [Epoch 1/200] [Batch 589/590] [D loss: 1.328426] [G loss: 0.732386] [recon loss: 0.171184]



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-5-fc3e5c733cc3> in <module>
         22         recon_loss = F.binary_cross_entropy(X_sample + TINY, real_imgs + TINY)
         23
    ---> 24         recon_loss.backward()
         25         P_decoder.step()
         26         Q_encoder.step()


    <decorator-gen-247> in backward(self, grad_tensor, retain_graph)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py in __impl__(func, *args, **kwargs)
         23     def __impl__(func, *args, **kwargs):
         24         wrapped_func = decorator_func(func)
    ---> 25         return wrapped_func(*args, **kwargs)
         26
         27     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py in __impl__(*args, **kwargs)
        225         assert in_dygraph_mode(
        226         ), "We only support '%s()' in dynamic graph mode, please call 'paddle.disable_static()' to enter dynamic graph mode." % func.__name__
    --> 227         return func(*args, **kwargs)
        228
        229     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in backward(self, grad_tensor, retain_graph)
        237             else:
        238                 core.dygraph_run_backward([self], [grad_tensor], retain_graph,
    --> 239                                           framework._dygraph_tracer())
        240         else:
        241             raise ValueError(


    KeyboardInterrupt:


## Step3: 评估Likelihood
原文中使用Parzen 窗估计方法.这是一种核密度估计的方法,具体可以参考

[英文介绍](http://https://sebastianraschka.com/Articles/2014_kernel_density_est.html)

[中文介绍](http://https://blog.csdn.net/Nianzu_Ethan_Zheng/article/details/79211861)

[GoodFellow 论文代码(需要Theano和Pylearn2)](http://https://github.com/goodfeli/adversarial/blob/master/parzen_ll.py)

此处实现放在`parzen_ll.py`中,使用numpy重写

以下为估计模型参数的Log-likelihood代码


```python
import pickle
import argparse

import paddle
from paddle.io import random_split
from paddle.io import DataLoader
import paddle.vision.transforms as transforms

from network import Encoder, Decoder, Discriminator
from utils.parzen_ll import *
from utils.log import get_logger
from config import args_parser

def load_data():
    trainset = paddle.load("./data/train")
    traindataloader = DataLoader(
        trainset, batch_size=opt.batchsize, shuffle=True,
    )
    valid_imgs = paddle.load("./data/valid")
    test_imgs = paddle.load("./data/test")

    return traindataloader,valid_imgs,test_imgs

if __name__ == "__main__":
    # Training settings
    opt = args_parser()

    logger = get_logger('./logs/eval.log')
    logger.info(opt)

    checkpoint = paddle.load("./model/model" + str(opt.load_epoch) + ".pkl")
    encoder = Encoder()
    decoder = Decoder()
    encoder_dict = checkpoint['encoder']
    decoder_dict = checkpoint['decoder']
    encoder.set_state_dict(encoder_dict)
    decoder.set_state_dict(decoder_dict)
    encoder.eval()
    decoder.eval()
    logger.info("model/model%d.pkl loaded!" % opt.load_epoch)

    # preprocessing
    z = paddle.normal(0,opt.std,(opt.N_gen, opt.latent_dim))
    traindataloader,valid_imgs,test_imgs = load_data()

    gen_imgs = decoder(z)
    train_imgs, _ = next(iter(traindataloader))

    gen_imgs = paddle.reshape(gen_imgs, (opt.N_gen, -1))
    train_imgs = paddle.reshape(train_imgs, (opt.batchsize, -1))
    valid_imgs = paddle.reshape(valid_imgs, (opt.N_valid, -1))
    test_imgs = paddle.reshape(test_imgs, (opt.N_test, -1))

    gen = np.asarray(gen_imgs.detach().cpu())
    train = np.asarray(train_imgs.detach().cpu())
    test = np.asarray(test_imgs.detach().cpu())
    valid = np.asarray(valid_imgs.detach().cpu())

    # cross validate sigma
    if opt.sigma is None:
        sigma_range = np.logspace(start = -1, stop = -0.3, num=20)
        sigma = cross_validate_sigma(
            gen, valid, sigma_range, batch_size = opt.batchsize, logger = logger
        )
        opt.sigma = sigma
    else:
        sigma = float(opt.sigma)
    logger.info("Using Sigma: {}".format(sigma))

    # fit and evaulate
    # gen_imgs
    parzen = parzen_estimation(gen, sigma)
    ll = get_nll(test, parzen, batch_size = opt.batchsize)
    se = ll.std() / np.sqrt(test.shape[0])
    logger.info("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
    ll = get_nll(valid, parzen, batch_size = opt.batchsize)
    se = ll.std() / np.sqrt(valid.shape[0])
    logger.info("Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se))

    logger.info("finish evaluation")


```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
