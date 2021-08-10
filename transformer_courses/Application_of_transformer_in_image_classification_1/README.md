# 基于 Swin Transformer 的图像分类 [[English](./README_en.md)]

## 依赖模块
- os
- numpy
- opencv
- pillow
- paddlepaddle==2.1.1

## 项目介绍
```
|-data: 存放ImageNet验证集
|-model_file:存放模型权重文件
|-transform.py: 数据预处理脚本
|-dataset.py: 读取数据脚本
|-swin_transformer.py: 该脚本中定义了Swin Transformer的网络结构
|-eval.py: 启动模型评估的脚本
```

**模型介绍**

Swin Transformer是一种新的视觉领域的Transformer模型，来自论文“Swin Transformer: Hierarchical Vision Transformer using Shifted Windows”，该模型可以作为计算机视觉任务的backbone。[论文地址](https://arxiv.org/pdf/2103.14030.pdf)

## 数据集准备
- 进入 repo 目录

  ```
  cd Swin_Transformer_for_image_classification
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

- 下载Swin Transformer的模型权重文件到`model_file`目录下

  ```
  mkdir model_file && cd model_file
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams
  cd ../
  ```

## 模型评估

可以通过以下方式开始模型评估过程

```bash
python3 eval.py 
    --model SwinTransformer  \
    --data data/ILSVRC2012_val
```

上述命令中，需要传入如下参数:

+ `model`: 模型名称;
+ `data`: 保存ImageNet验证集的目录, 默认值为 `data/ILSVRC2012_val`。