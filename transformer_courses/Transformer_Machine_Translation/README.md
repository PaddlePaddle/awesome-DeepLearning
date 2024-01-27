# 基于 Transformer 的机器翻译 [[English](./README_en.md)]

## 依赖模块
- python3
- subword_nmt==0.3.7
- attrdict==2.0.1
- paddlenlp==2.0.0rc22


## 项目介绍
```
|-data_process.py: 数据集预处理
|-bpe_process.py: jieba分词
|-bpe_process2.py: bpe数据预处理脚本
|-dataloader.py: dataloader迭代器脚本
|-train.py: 该脚本中定义了Transformer的训练
|-predict.py: 启动模型预测的脚本
```

**模型介绍**

Transformer 是 Google 团队在 17 年 6 月提出的 NLP 经典之作，由 Ashish Vaswani 等人在 2017 年发表的论文 Attention Is All You Need 中提出。Transformer 在机器翻译任务上的表现超过了 RNN，CNN，只用 encoder-decoder 和 attention 机制就能达到很好的效果，最大的优点是可以高效地并行化。[论文地址](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)

## 安装依赖

- 进入 repo 目录

  ```
  cd Transformer_Machine_Translation
  ```
- 安装依赖

  ```
  pip install -r requirements.txt
  ```

## 数据集准备

- 下载[2015-01.tgz数据集](https://wit3.fbk.eu/2015-01)到根目录下

  ```
  tar -xvf 2015-01.tgz
  tar -xvf 2015-01/texts/zh/en/zh-en.tgz
  cd path_to_Transformer-classification
  ```

## bpe分词处理

  ```
  python data_process.py
  sh subword.sh
  python bpe_process2.py
  ```
## 模型准备

- 进入 repo 目录

  ```bash
  cd path_to_Transformer-classification
  ```

  ```bash
  python train.py
  ```

## 模型评估

可以通过以下方式开始模型评估过程

  ```bash
  python predict_py.py
  ```