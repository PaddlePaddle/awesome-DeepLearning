# 基于paddle实现的ViLBERT模型

基于[paddle](https://github.com/PaddlePaddle/Paddle)框架的[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)实现

**注:本项目根目录在/home/aistudio/work/ViLBERT-REC-Paddle下**

## 一、简介

本项目使用[paddle](https://github.com/PaddlePaddle/Paddle)框架复现[ViLBERT](https://arxiv.org/abs/1908.02265)模型。该模型包含两个并行的流，分别用于编码visual和language，并且加入了 co-attention transformer layer 来增强两者之间的交互，得到 pretrained model。作者在多个 vision-language 任务上得到了多个点的提升。

**论文:**

* [1] J. Lu, D. Batra, D. Parikh, S. Lee, "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks", NIPS, 2019.

**参考项目:**

* [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task) [官方实现]

## 二、复现精度

> 所有指标均为模型在[RefCOCO+](https://arxiv.org/abs/1608.00272)的验证集评估而得

| 指标 | 原论文 | 复现精度 |
| :---: | :---: | :---: |
| Acc | 72.34 | 72.71 |

## 三、数据集

本项目所使用的数据集为[RefCOCO+](https://arxiv.org/abs/1608.00272)。该数据集共包含来自19,992张图像的49,856个目标对象，共计141,565条指代表达。本项目使用作者提供的预提取的`bottom-up`特征，可以从[这里](https://www.dropbox.com/sh/4jqadcfkai68yoe/AADHI6dKviFcraeCMdjiaDENa?dl=0)下载得到（我们提供了脚本下载该数据集以及图像特征，见[download_dataset.sh](https://github.com/fuqianya/ViLBERT-Paddle/blob/main/download_dataset.sh)）。

## 四、环境依赖

* 硬件：CPU、GPU

* 软件：
    * Python 3.8
    * PaddlePaddle == 2.1.0

## 五、快速开始

### Step1: 安装环境依赖


```python
!pip install boto3 lmdb
```

### Step2: 下载数据

```bash
# 下载数据集
bash ./download_dataset.sh

# 下载paddle格式的预训练模型
# 放于checkpoints/bert_base_6_layer_6_connect_freeze_0下
# 下载链接: https://drive.google.com/file/d/1QMJz5anz_git8NFThUgacOBgYti_of4g/view?usp=sharing

# 编译REFER
cd pyutils/refer && make
cd ..
```

### Step3: 训练


```python
!cd /home/aistudio/work/ViLBERT-REC-Paddle && python train.py
```

    09/06/2021 14:23:14 - INFO - __main__ -   Loading refcoco+ Dataset with batch size 256
    [32m[2021-09-06 14:23:14,931] [    INFO][0m - Downloading bert-base-uncased-vocab.txt from https://paddle-hapi.bj.bcebos.com/models/bert/bert-base-uncased-vocab.txt[0m
    100%|███████████████████████████████████████| 227/227 [00:00<00:00, 4258.58it/s]
    loading dataset refcoco+ into memory...
    creating index...
    index created.
    DONE (t=12.26s)
    42278 refs are in split [train].
    loading entries from data/referExpression/cache/refcoco+_train_20_100.pkl
    loading dataset refcoco+ into memory...
    creating index...
    index created.
    DONE (t=15.59s)
    3805 refs are in split [val].
    loading entries from data/referExpression/cache/refcoco+_val_20_100.pkl
    W0906 14:23:45.809420   289 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0906 14:23:45.813421   289 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    ***** Running training *****
      Num Iters:  470
      Batch size:  256
      Num steps: 9400
    ====>start epoch 0:
    Training: 100%|███████████████████████████████| 470/470 [24:08<00:00,  3.08s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:43<00:00,  1.01s/it]
    09/06/2021 14:48:47 - INFO - __main__ -   ** ** Epoch {0} done! Traing loss: 3.83981, Val accuracy: 0.6459
    ====>start epoch 1:
    Training: 100%|███████████████████████████████| 470/470 [23:51<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.00it/s]
    09/06/2021 15:13:23 - INFO - __main__ -   ** ** Epoch {1} done! Traing loss: 2.31176, Val accuracy: 0.6949
    09/06/2021 15:13:23 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 2:
    Training: 100%|███████████████████████████████| 470/470 [23:54<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.03it/s]
    09/06/2021 15:38:03 - INFO - __main__ -   ** ** Epoch {2} done! Traing loss: 1.80026, Val accuracy: 0.7080
    ====>start epoch 3:
    Training: 100%|███████████████████████████████| 470/470 [23:56<00:00,  3.06s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.04it/s]
    09/06/2021 16:02:41 - INFO - __main__ -   ** ** Epoch {3} done! Traing loss: 1.40603, Val accuracy: 0.7086
    09/06/2021 16:02:41 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 4:
    Training: 100%|███████████████████████████████| 470/470 [23:55<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.02it/s]
    09/06/2021 16:27:22 - INFO - __main__ -   ** ** Epoch {4} done! Traing loss: 1.13971, Val accuracy: 0.7107
    ====>start epoch 5:
    Training: 100%|███████████████████████████████| 470/470 [23:55<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:40<00:00,  1.05it/s]
    09/06/2021 16:51:59 - INFO - __main__ -   ** ** Epoch {5} done! Traing loss: 0.96396, Val accuracy: 0.7120
    09/06/2021 16:51:59 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 6:
    Training: 100%|███████████████████████████████| 470/470 [23:51<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.04it/s]
    09/06/2021 17:16:35 - INFO - __main__ -   ** ** Epoch {6} done! Traing loss: 0.86292, Val accuracy: 0.7156
    ====>start epoch 7:
    Training: 100%|███████████████████████████████| 470/470 [23:57<00:00,  3.06s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:40<00:00,  1.07it/s]
    09/06/2021 17:41:13 - INFO - __main__ -   ** ** Epoch {7} done! Traing loss: 0.78510, Val accuracy: 0.7173
    09/06/2021 17:41:13 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 8:
    Training: 100%|███████████████████████████████| 470/470 [23:52<00:00,  3.05s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:40<00:00,  1.06it/s]
    09/06/2021 18:05:49 - INFO - __main__ -   ** ** Epoch {8} done! Traing loss: 0.72758, Val accuracy: 0.7139
    ====>start epoch 9:
    Training: 100%|███████████████████████████████| 470/470 [23:57<00:00,  3.06s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.03it/s]
    09/06/2021 18:30:29 - INFO - __main__ -   ** ** Epoch {9} done! Traing loss: 0.69806, Val accuracy: 0.7183
    09/06/2021 18:30:29 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 10:
    Training: 100%|███████████████████████████████| 470/470 [23:40<00:00,  3.02s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.02it/s]
    09/06/2021 18:54:56 - INFO - __main__ -   ** ** Epoch {10} done! Traing loss: 0.66901, Val accuracy: 0.7122
    ====>start epoch 11:
    Training: 100%|███████████████████████████████| 470/470 [23:47<00:00,  3.04s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.03it/s]
    09/06/2021 19:19:26 - INFO - __main__ -   ** ** Epoch {11} done! Traing loss: 0.64070, Val accuracy: 0.7201
    09/06/2021 19:19:26 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 12:
    Training: 100%|███████████████████████████████| 470/470 [23:44<00:00,  3.03s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.02it/s]
    09/06/2021 19:43:56 - INFO - __main__ -   ** ** Epoch {12} done! Traing loss: 0.57114, Val accuracy: 0.7263
    ====>start epoch 13:
    Training: 100%|███████████████████████████████| 470/470 [23:42<00:00,  3.03s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.02it/s]
    09/06/2021 20:08:22 - INFO - __main__ -   ** ** Epoch {13} done! Traing loss: 0.54065, Val accuracy: 0.7264
    09/06/2021 20:08:22 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 14:
    Training: 100%|███████████████████████████████| 470/470 [23:42<00:00,  3.03s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.03it/s]
    09/06/2021 20:32:49 - INFO - __main__ -   ** ** Epoch {14} done! Traing loss: 0.52415, Val accuracy: 0.7254
    ====>start epoch 15:
    Training: 100%|███████████████████████████████| 470/470 [23:44<00:00,  3.03s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.01it/s]
    09/06/2021 20:57:16 - INFO - __main__ -   ** ** Epoch {15} done! Traing loss: 0.51826, Val accuracy: 0.7271
    09/06/2021 20:57:16 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 16:
    Training: 100%|███████████████████████████████| 470/470 [23:46<00:00,  3.03s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.01it/s]
    09/06/2021 21:21:48 - INFO - __main__ -   ** ** Epoch {16} done! Traing loss: 0.51339, Val accuracy: 0.7273
    ====>start epoch 17:
    Training: 100%|███████████████████████████████| 470/470 [23:46<00:00,  3.04s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:42<00:00,  1.02it/s]
    09/06/2021 21:46:18 - INFO - __main__ -   ** ** Epoch {17} done! Traing loss: 0.51202, Val accuracy: 0.7272
    09/06/2021 21:46:18 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    ====>start epoch 18:
    Training: 100%|███████████████████████████████| 470/470 [23:48<00:00,  3.04s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.04it/s]
    09/06/2021 22:10:51 - INFO - __main__ -   ** ** Epoch {18} done! Traing loss: 0.50981, Val accuracy: 0.7274
    ====>start epoch 19:
    Training: 100%|███████████████████████████████| 470/470 [23:49<00:00,  3.04s/it]
    Eval: 100%|█████████████████████████████████████| 43/43 [00:41<00:00,  1.03it/s]
    09/06/2021 22:35:23 - INFO - __main__ -   ** ** Epoch {19} done! Traing loss: 0.50994, Val accuracy: 0.7271
    09/06/2021 22:35:23 - INFO - __main__ -   ** ** * Saving fine - tuned model on bert_base_6layer_6conect-refcoco+** ** *
    [0m

### Step4: 测试


```python
!python eval.py --from_pretrained ./checkpoints/refcoco+_bert_base_6layer_6conect-pretrained/paddle_model_19.pdparams
```

### 使用预训练模型进行预测

模型下载: [谷歌云盘](https://drive.google.com/file/d/19gbGuVm9hgVPm_XzAUrTpeDmObr5ZAv3/view?usp=sharing)

将下载的模型权重以及训练信息放到`checkpoints/refcoco+_bert_base_6layer_6conect-pretrained`目录下, 运行`step5`的指令进行测试。

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
