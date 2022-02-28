# 前言

本项目为百度论文复现第四期《Only a Matter of Style: Age Transformation Using a Style-Based Regression Model》论文复现第一名代码以及模型。

官方源码：[https://github.com/yuval-alaluf/SAM](https://github.com/yuval-alaluf/SAM)

复现地址：[https://github.com/771979972/paddle-SAM](https://github.com/771979972/paddle-SAM)

# 应用领域：图像编辑
Age Transformationation是指将一个人的图片变化为不同年龄的样子，同时还要保持自己身份的过程。最近，随着越来越多的app允许用户进行面部编辑，年龄转换这项任务受到了越来越多的关注。

# 技术方向
生成对抗网络；Age Transformation；图像编辑

# 模型结构
![](https://ai-studio-static-online.cdn.bcebos.com/0a52084def3944dc844997e5ecd6ca9472f96906537e4575b60130399f94cd41)


作者提出将age transform视为I2I任务，将预训练、固定参数的StyleGAN生成器和psp encoder组合在一起。在训练过程中，encoder被输入一张图片，并将他们映射到要求的年龄所在的潜在空间得到一个潜在向量，这个潜在向量通过styleGAN生成最终的图片。这种方法利用了预训练模型，极大地降低了训练难度和时间成本。这个模型被称为SAM。

实现细节：训练中，向SAM模型输入人脸图片和目标年龄𝛼𝑡，年龄编码器提取三个不同大小的特征图，然后通过18个map2style块生成18个512维的向量。接着，预训练的psp编码器将图片编码成潜在向量，然后二者相加，传入StyleGAN生成目标图片。

比起依赖于预定义年龄组的multi-domain和anchor classes方法，他们将人类衰老看作是一个连续的回归过程，从而可以对转换进行细粒度的控制。SAM不依赖于标签好年龄的数据，而是使用了预训练模型来判断年龄。

# 结果
目前呈现的结果为运行24000步保存的模型的结果，据作者称论文的结果为运行了60000步.

图片从左到右分别是：输入图片，SAM模型依次生成0岁，10岁，20岁，30岁，40岁，50岁，60岁，70岁，80岁，90岁，100岁图片

### Pytorch与Paddle效果对比
| 模型 | 图片 |
| ------ | ------ |
| Pytorch |  ![](https://ai-studio-static-online.cdn.bcebos.com/f694aa85db1f41b99685aa74984512f7f5ffadd289ab40bbae253b77572e3d44)|
| Paddle | ![](https://ai-studio-static-online.cdn.bcebos.com/bbd4c8b5d7624acfa74280f237a2160502e5834063c84f008019d6212351d096) |
| Pytorch | ![](https://ai-studio-static-online.cdn.bcebos.com/01e35228b4ca451f9f58091a374de6049eb68b4f92bf4b1a8f483317db6f56a3) |
| Paddle | ![](https://ai-studio-static-online.cdn.bcebos.com/2a727e5efa5a45aa86cbcd4cd375d5a849dd8d7fff244f7e9fa1e7a65de72dba) |
| Pytorch |![](https://ai-studio-static-online.cdn.bcebos.com/47dfbd675ae141e4b9cc10ed8c7b39413ffd9e46ffaa44d0bbc6a21684f1e413) |
| Paddle | ![](https://ai-studio-static-online.cdn.bcebos.com/51b2799fdf2e45fba39d4bf2b7f7959d9092cea658824fffa52278449df08646) |

### 以下是使用Paddle复现的其他结果
![](https://ai-studio-static-online.cdn.bcebos.com/a787dca78db541048a00e297c3d892fd60a409be090c4675bdbf76f6c368349e)
![](https://ai-studio-static-online.cdn.bcebos.com/ae7679e1ef2641f1984d4422881ea0fb25877394ef9a42f29dd34a5e72a00ccc)
![](https://ai-studio-static-online.cdn.bcebos.com/793ef44b3e8b4c5997046fa954e93262b44b59945f4642e19c7de79989ee72b9)

## Dataset
训练集解压：FFHQ-512（已挂载在AI studio项目中）。解压后保存在```work\FFHQ\```。

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
解压paddle-sam.zip到```work\```

然后执行```cd work```

### Inference
```
python scripts/inference_side_by_side.py
--exp_dir=exp/test
--checkpoint_path=pretrained_models/sam_ffhq_aging.pdparams
--data_path=CelebA_test
--test_batch_size=4
--test_workers=0
--target_age=0,10,20,30,40,50,60,70,80,90,100
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
