# 基于预训练模型 ERNIE-Gram 实现语义匹配

本案例介绍 NLP 最基本的任务类型之一 —— 文本语义匹配，并且基于 PaddleNLP 使用百度开源的预训练模型 ERNIE-Gram 搭建效果优异的语义匹配模型，来判断 2 段文本语义是否相同。


# 一. 背景介绍
文本语义匹配任务，简单来说就是给定两段文本，让模型来判断两段文本是不是语义相似。

在本案例中以权威的语义匹配数据集 [LCQMC](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) 为例，[LCQMC](https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition) 数据集是基于百度知道相似问题推荐构造的通问句语义匹配数据集。训练集中的每两段文本都会被标记为 1（语义相似） 或者 0（语义不相似）。更多数据集可访问[千言](https://www.luge.ai/)获取哦。

例如百度知道场景下，用户搜索一个问题，模型会计算这个问题与候选问题是否语义相似，语义匹配模型会找出与问题语义相似的候选问题返回给用户，加快用户提问-获取答案的效率。例如，当某用户在搜索引擎中搜索 “深度学习的教材有哪些？”，模型就自动找到了一些语义相似的问题展现给用户:
![](https://ai-studio-static-online.cdn.bcebos.com/ecc1244685ec4476b869ce8a32d421c0ad530666e98d487da21fa4f61670544f)

本项目AI Studio版本请参考：[https://aistudio.baidu.com/aistudio/projectdetail/2535083](https://aistudio.baidu.com/aistudio/projectdetail/2535083)

# 二、方案设计


<center><img src='https://ai-studio-static-online.cdn.bcebos.com/8d8cc66af43c4dd0ad9e3e26e1deba17ed4184cf3d8e459cb28e5f0699e756e9' width='700px'></center>

query代表用户的请求，title代表是待匹配的文本，query和title转换成id后，经过Ernie-Gram模型，得到模型的输出，输出即为query和title的相似度。

# 三、 数据处理

## 3.1 数据加载

为了训练匹配模型，一般需要准备三个数据集：训练集 train.tsv、验证集dev.tsv、测试集test.tsv。此案例我们使用 PaddleNLP 内置的语义数据集 [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 来进行训练、评估、预测。

训练集: 用来训练模型参数的数据集，模型直接根据训练集来调整自身参数以获得更好的分类效果。

验证集: 用于在训练过程中检验模型的状态，收敛情况。验证集通常用于调整超参数，根据几组模型验证集上的表现，决定采用哪组超参数。

测试集: 用来计算模型的各项评估指标，验证模型泛化能力。

[LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 数据集是公开的语义匹配权威数据集。PaddleNLP 已经内置该数据集，一键即可加载。

安装相应环境下最新版飞桨框架。使用如下命令确保安装最新版PaddleNLP：


```bash
pip install --upgrade paddlenlp
```

# 四、 模型训练


```
python train.py
```


# 五、模型预测

# 下载我们基于 Lcqmc 事先训练好的语义匹配模型并解压

```
wget https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
tar -xvf ernie_gram_zh_pointwise_matching_model.tar

```

```
python3 predict.py
```



**数据来源**

本案例数据集来源于：https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition
