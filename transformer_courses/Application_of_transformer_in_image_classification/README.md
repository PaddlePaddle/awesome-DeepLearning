# 基于 Transformer 的图像分类 [[English](./README_en.md)]

## 依赖模块
- os
- numpy
- opencv
- pillow
- paddlepaddle==2.0.0

## 项目介绍
```
|-data: 存放ImageNet验证集
|-model_file:存放模型权重文件
|-transform.py: 数据预处理脚本
|-dataset.py: 读取数据脚本
|-model.py: 该脚本中定义了ViT以及DeiT的网络结构
|-eval.py: 启动模型评估的脚本
```

当前项目仅支持对ViT和DeiT进行模型评估，模型训练过程会在后续更新中进行支持。

**模型介绍**

ViT（Vision Transformer）系列模型是Google在2020年提出的，该模型仅使用标准的Transformer结构，完全抛弃了卷积结构，将图像拆分为多个patch后再输入到Transformer中，展示了Transformer在CV领域的潜力。[论文地址](https://arxiv.org/abs/2010.11929)

DeiT（Data-efficient Image Transformers）系列模型是由FaceBook在2020年底提出的，针对ViT模型需要大规模数据集训练的问题进行了改进，最终在ImageNet上取得了83.1%的Top1精度。并且使用卷积模型作为教师模型，针对该模型进行知识蒸馏，在ImageNet数据集上可以达到85.2%的Top1精度。[论文地址](https://arxiv.org/abs/2012.12877)

## 数据集准备
- 进入 repo 目录

  ```
  cd path_to_Transformer-classification
  ```

- 下载[ImageNet验证集](https://aistudio.baidu.com/aistudio/datasetdetail/93561)并解压到`data`目录下

  ```
  mkdir data && cd data
  tar -xvf ILSVRC2012_val.tar
  cd ../
  ```

- 请按照如下格式组织数据集

  ```
  data/ILSVRC2012_val
  |_ val
  |_ val_list.txt
  ```

## 模型准备

- 下载ViT和DeiT的模型权重文件到`model_file`目录下

  ```
  mkdir model_file && cd model_file
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_384_pretrained.pdparams
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams
  cd ../
  ```

## 模型评估

可以通过以下方式开始模型评估过程

```bash
python3 eval.py 
    --model ViT  \
    --data data/ILSVRC2012_val
```

上述命令中，需要传入如下参数:

+ `model`: 模型名称，默认值为 `ViT`，可以更换为 `DeiT`;
+ `data`: 保存ImageNet验证集的目录, 默认值为 `data/ILSVRC2012_val`。