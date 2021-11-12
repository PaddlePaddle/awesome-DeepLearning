## 前言

本项目为百度论文复现第四期《[ Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation](https://paperswithcode.com/paper/encoding-in-style-a-stylegan-encoder-for) 》论文复现第一名代码。

官方源码：[https://github.com/eladrich/pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

paddle复现地址：[https://raw.githubusercontent.com/771979972/Paddle_pSp/](https://raw.githubusercontent.com/771979972/Paddle_pSp/)

## 应用领域
近年来，生成对抗网络(GANs)具有先进的图像合成效果。最前沿的方法已经实现了较高的视觉质量和保真度，可以生成具有惊人真实性的图像。比如StyleGAN。
StyleGAN提出了一种新的生成器架构，并在高分辨率图像生成任务上获得极佳的视觉质量。此外，已经有研究证明了它有一个解纠缠的潜在空间——W空间，它提供了控制和编辑图像的能力。通过修改在W空间的潜在向量，可以辅助生成不同样子的图片。
pixel2style2pixel encoder介绍了一种新的编码器架构，即将任意图像直接编码到W+空间中。该编码器基于特征金字塔网络，其中样式向量从不同的金字塔尺度中提取，并直接插入到一个固定的、预先训练过的StyleGAN生成器中，以对应于它们的空间尺度。

## 技术方向
生成对抗网络；GAN Inversion

## 模型结构
![](https://ai-studio-static-online.cdn.bcebos.com/51420eb76bef4242a5c00a028acb95b013a8adb3fc074883bd02b8f147a77b3a)
模型名为pSp，即pixel2style2pixel。它首先将特征图通过一个基于ResNet的特征金字塔结构来提取出特征图。特征图有三个，分别是small, medium, large。对于这些特征图，pSp训练了一个小型的映射网络，从small特征图提取style0-2，medium提取style3-6，large提取style7-18，然后再传入StyleGAN的Affine transformation，最后生成图片。


## 评估指标（在CelebA-HQ上测试）：
以下三行数据分别是论文所述StyleGAN inversion任务指标，使用官方pytorch模型在CelebA-HQ实际跑出来的指标，以及paddle模型跑出来的指标.
| 模型 | LPIPS | Similarity | MSE |
| ------ | ------ | ------ | ------ |
| 论文 | 0.17 | 0.56 | 0.03 |
| Pytorch模型 | 0.15 | 0.57 | 0.03 |
| Paddle模型 | 0.17 | 0.57 | 0.03 |

## Dataset
训练集解压：FFHQ-1024（在AI studio上搜索FFHQ1024下挂载数据集）。解压后保存在```work\FFHQ\```。

测试集解压：CelebA-HQ（已挂载在AI studio项目中）。下载后将val中的female和male两个文件夹的图片数据保存```work\CelebA_test\```

## 预训练模型：
下载后将模型的参数保存在```work\pretrained_models\```中
| 模型(文件名) | Description
| :--- | :----------
|FFHQ StyleGAN(stylegan2-ffhq-config-f.pdparams) | StyleGAN 在FFHQ上训练，来自 [rosinality](https://github.com/rosinality/stylegan2-pytorch) ，输出1024x1024大小的图片
|IR-SE50 Model(model_ir_se50.pdparams) | IR SE 模型，来自 [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) 用于训练中计算ID loss。
|CurricularFace Backbone(CurricularFace_Backbone.paparams)  | 预训练的 CurricularFace model，来自 [HuangYG123](https://github.com/HuangYG123/CurricularFace) 用于Similarity的评估。
|AlexNet(alexnet.pdparams和lin_alex.pdparams)  | 用于lpips loss计算。
|StyleGAN Inversion(psp_ffhq_inverse.pdparams)  | pSp trained with the FFHQ dataset for StyleGAN inversion.|

链接：[https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg](https://pan.baidu.com/s/1G-Ffs8-y93R0ZlD9mEU6Eg )
提取码：m3nb

## 具体使用
解压**paddle-psp.zi**p到```work\```

然后执行```cd work```

### Inference
```
python scripts/inference.py \
--exp_dir=inference \
--checkpoint_path=pretrained_models/psp_ffhq_inverse.pdparams \
--data_path=CelebA_test \
--test_batch_size=8 \
--test_workers=4
```

### 训练
首先配置环境
```
!pip install --upgrade matplotlib
python scripts/compile_ranger.py
```
然后再训练
```
python scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=exp/test \
--workers=0 \
--batch_size=8 \
--test_batch_size=8 \
--test_workers=0 \
--val_interval=2500 \
--save_interval=5000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1 \
--id_lambda=0.1 \
--optim_name=ranger
```
#### 其他指标的计算：
计算LPIPS
```
python scripts/calc_losses_on_images.py \
--mode lpips \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```
计算MSE
```
python scripts/calc_losses_on_images.py \
--mode l2 \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```
计算Similarity
```
python scripts/calc_id_loss_parallel.py \
--data_path=inference/inference_results \
--gt_path=CelebA_test
```
