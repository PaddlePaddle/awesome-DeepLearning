# 前言
本项目为百度论文复现第四期《Deep image prior》论文复现第一名代码以及模型。 依赖环境：

* paddlepaddle-gpu 2.1.2
* python 3.7

数据集： Set14 数据集： https://github.com/jbhuang0604/SelfExSR/tree/master/data

验收标准： 8 × super-resolution, avg psnr = 24.15%

参考代码： https://github.com/DmitryUlyanov/deep-image-prior

# 论文简介
传统观点认为，模型需要从大量真实图像数据中学习图像的先验信息，再通过这些学到的先验信息来修补图像或生成高分辨率图像。

然而该论文的作者认为，自然图像具有很强的局部规律和自相似性，让一个卷积生成器在单个图像上反复迭代训练，甚至是在破损的图像上训练，同样能学习到图像的先验信息，并可利用该图像自身的先验信息来完成去噪、去水印、超分辨率等功能。

作者在摘要中指出：

> In this paper, we show that, on the contrary, the *structure* of a generator network is sufficient to capture a great deal of low-level image statistics *prior to any learning*.

此外，文中指出：

>In this work, we show that, in fact, not all image priors must be learned from data; instead, a great deal of  image statistics are captured by the structure of generator ConvNets, independent of learning.

也就是说，大量的图像的统计信息事实上是由卷积生成网络提取出来的，而不是通过从数据中学习得到的。

作者基于这一理论，采用没有训练的卷积生成网络来处理图像实现去噪等任务。由于网络权重是随机的，所以唯一的先验信息是来自**网络结构本身**。

此外，作者发现

> In particular, we show that the network resists “bad” solutions and descends much more quickly towards naturally-looking images.

即，网络对图片特征的学习会对“坏”的解具备高阻抗特性，即网络会优先学习更加“自然”的特征.

# 算法实现

## 损失函数
无论是图像去噪还是修复，通常都可以用以下目标函数描述：

$$x^*=\mathop {argmin}_{x} E(x;x_0)+R(x),\tag{1}$$

其中$x_0$​​​​​​是原始图像，$x^*$​​​​​​是最后的输出图像，$E(x;x_0)$​​​​​​是一个任务依赖的函数，$R(x)$​​​​​​是一个正则化项。根据不同的任务可以选择不同的$E$​​​​​​，例如去噪可以选用MSE，即二范数${\Vert x-x_0 \Vert}^2$​​​​，正则化项可以选择TV项。

然而论文中，作者摒弃了显式的正则化项$R(x)$，而使用神经网络参数化捕获的隐式先验：

$$\theta^*=\mathop {argmin}_{\theta} E(f_{\theta}(z);x_0),\qquad x^*=f_{\theta^*}(z)\tag{2}$$

其中$\theta^*$为随机初始化参数$\theta$，经训练得到的（局部）最小值，$z$是输入网络的编码，为固定的一个随机初始化的3D张量，$x^*$​为最后的输出图像。

## 算法流程：
1. 选择一个深度卷积生成网络，通常选择具有编码-解码结构的网络，随机初始化参数；
2. 随机初始化一个编码 $z$​​​ ，固定其作为网络的输入，该编码与图像的大小一致，通道数可以不一致；
3. 目标函数通常选用MSE。对于去噪问题，输出图像与原始图像的MSE最小；对于超分辨率问题，为输出的高分辨率图像的下采样与原始图像间的MSE最小；
4. 每次迭代时，对输入加入一个扰动，即对编码 $z$​​​​ 添加一个均值为0，方差为$\sigma_p$​​​​的高斯白噪声；
5. 采用Adam方法训练，待结果稳定或到预设的最大迭代次数后，停止训练；
6. 训练结束后，输入训练时输入的编码 $z$​ ，即可得到最终的输出结果。

## 网络框架
作者采用Hourglass网络，本项目重写了该网络，并调整了整体代码结构，使得更清晰明了

# 运行项目

## 依赖
在Ai studio上运行的话，首先需要安装`scikit-image`，否则无法显示图像


```python
!pip install scikit-image
```

## 导入所需库


```python
from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import argparse
import os
from PIL import Image
import PIL
import numpy as np
from models import *

import paddle
from models.downsampler import Downsampler

from utils.sr_utils import *
from utils.common_utils import *
```

## 设置超参数

其中`factor`为缩放大小，本实验为实验8x超分辨，4x可采用`factor=4`时的相关设置


```python
input_depth = 32

INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

# factor == 4:
# num_iter = 2000
# reg_noise_std = 0.03

factor = 8
num_iter = 5000
reg_noise_std = 0.05

imsize = -1
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True

NET_TYPE = 'skip'
mse = nn.MSELoss()
```

## 导入数据&训练


```python
file_names = ["baboon", "barbara", "bridge", "coastguard", "comic", "face", "flowers", "foreman", "lenna", "man", "monarch", "pepper", "ppt3", "zebra"]
data_dir = 'datas/sr/Set14/'

psnr_DIP = []
psnr_all = []
```


```python
for name in file_names:
    # Load Data
    path = data_dir + name
    img = load_LR_HR_imgs_sr(path + '_GT.png', imsize, factor, enforse_div32)
    img['bicubic_np'], img['sharp_np'], img['nearest_np'] = get_baselines(img['LR_pil'], img['HR_pil'])

    img_LR_var = paddle.to_tensor(img['LR_np']).unsqueeze(0)
    n_channels = img['LR_np'].shape[0]

    # Get Net and Input
    net = get_net(input_depth, NET_TYPE, pad,
                  n_channels=n_channels,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear')
    downsampler = Downsampler(n_planes=n_channels, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True)
    net_input_z = get_noise(input_depth, INPUT, (img['HR_pil'].size[1], img['HR_pil'].size[0]))

    optimizer = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    psnr_history = []
    for i in range(num_iter):

        if reg_noise_std > 0:
            net_input = net_input_z + (paddle.normal(shape=net_input_z.shape) * reg_noise_std)
        else:
            net_input = net_input_z

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        # Log
        psnr_LR = compare_PSNR(np.clip(out_LR.squeeze(0).numpy(), 0, 1), img['LR_np'])
        psnr_HR = compare_PSNR(np.clip(out_HR.squeeze(0).numpy(), 0, 1), img['HR_np'])
        # print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')

        # History
        psnr_history.append([psnr_LR, psnr_HR])

        if PLOT and i % 1000 == 0:
            out_HR_np = out_HR.squeeze().numpy()
            plot_image_grid([img['HR_np'], img['bicubic_np'], np.clip(out_HR.squeeze(0).numpy(), 0, 1)], factor=13, nrows=3)

    # Results
    psnr_all.append(psnr_history)

    out_DIP = np.clip(net(net_input_z).squeeze(0).numpy(), 0, 1)
    psnr_out = compare_PSNR(out_DIP, img['HR_np'])
    print(name + ': PSNR_HR = %.3f' % (psnr_out))

    psnr_DIP.append(psnr_out)
    if n_channels==3:
        matplotlib.image.imsave(path + '_DIP.png', out_DIP.transpose(1,2,0))
    else:
        matplotlib.image.imsave(path + '_DIP.png', out_DIP.squeeze(), cmap = matplotlib.cm.gray)


```


```python
psnr_avg = np.mean(psnr_DIP)
```


```python
psnr_avg
```

## 结果

最终结果可在 datas/sr/Set14中查看


```python

```
