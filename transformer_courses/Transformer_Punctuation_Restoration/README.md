# 基于 ELECTRA 的标点符号预测 [[English](./README_en.md)]

## 依赖模块

- python3
- paddlenlp==2.0.0rc22 
- paddlepaddle==2.1.1
- pandas
- attrdict==2.0.1
- ujson
- tqdm
- paddlepaddle-gpu 

## 项目介绍

```
|-data_transfer.py: 将测试集和训练集数据从xml格式提取成txt形式
|-data_process.py: 数据集预处理，并且分别构建训练和测试数据集 
|-dataloader.py: 包含构建dataloader的方法
|-train.py: 构建dataloader，加载预训练模型，设置AdamW优化器，cross entropy损失函数以及评估方式该脚本中，并且定义了ELECTRA的训练
|-predict.py: 启动模型预测的脚本，并且储存预测结果于txt文件
```

**模型介绍**

ELECTRA 是由 Kevin Clark 等人（Standfold 和 Google 团队）在 ICLR 2020 发表的论文 ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS 中提出。其最大的贡献是提出了新的预训练任务 Replaced Token Detection (RTD) 和框架 ELECTRA。ELECTRA的RTD任务比MLM的预训练任务好，推出了一种十分适用于NLP的类GAN框架。最大的优点是，在现有的算力资源上，设计出更高效的模型结构与自监督预训练任务。[论文地址](https://arxiv.org/abs/2003.10555)

**任务介绍**

本实验采用的是Discriminator来做标点符号预测任务（punctuation restoration）。标点符号预测本质上是一种序列标注任务。本实验预测的标点符号有逗号，句号，问号3种。如果读者有兴趣，也可以把其他类型的标点符号加进去。

## 安装依赖

- 进入 repo 目录

  ```
  cd Transformer_Punctuation_Restoration
  ```
- 安装依赖

  ```
  pip install -r requirements.txt
  ```

## 数据集准备

- 下载[IWSLT12.zip数据集](https://aistudio.baidu.com/aistudio/datasetdetail/98318)并解压到`data`目录下

  ```
  mkdir data && cd data
  unzip IWSLT12.zip
  cd ../
  ```

- 请按照如下格式组织数据集

  ```
  data 
  |_ IWSLT12.TED.MT.tst2011.en-fr.en.xml
  |_ IWSLT12.TED.SLT.tst2011.en-fr.en.system0.comma.xml
  |_ IWSLT12.TALK.dev2010.en-fr.en.xml
  |_ IWSLT12.TED.MT.tst2012.en-fr.en.xml
  |_ train.tags.en-fr.en.xml
  ```

## 数据预处理

  ```bash
  python data_transfer.py  
  python data_process.py  
  ``` 

## 模型训练与评估

- 使用`electra.base.yaml`配置训练超参数后，进入模型训练。训练完成后对模型进行评估。
- 进入 repo 目录

  ```bash
  cd Transformer_Punctuation_Restoration
  ```

  ```bash
  python train.py
  ```

## 模型预测

- 选择`checkpoint`中的模型参数，在`electra.base.yaml`中配置，我们便可以通过以下方式开始模型对测试集的预测。最终预测出结果可以输出到txt文件中。

  ```bash
  python predict_py.py
  ```