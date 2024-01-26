# 代码运行情况

代码在paddle上是能够正常运行的，这是提交的链接https://aistudio.baidu.com/aistudio/projectdetail/2259710
使用了paddle上所配置的环境，电脑上直接运行的话跑不通，但是在paddle平台上是能够运行得到结果的。

# RNN简介

循环神经网络（Recurrent Neural Network, RNN）是一类以序列（sequence）数据为输入，在序列的演进方向进行递归（recursion）且所有节点（循环单元）按链式连接的递归神经网络（recursive neural network）。

它与DNN,CNN不同的是: 它不仅考虑前一时刻的输入,而且赋予了网络对前面的内容的一种'记忆'功能.

RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

对循环神经网络的研究始于二十世纪80-90年代，并在二十一世纪初发展为深度学习（deep learning）算法之一，其中双向循环神经网络（Bidirectional RNN, Bi-RNN）和长短期记忆网络（Long Short-Term Memory networks，LSTM）是常见的循环神经网络。

循环神经网络具有记忆性、参数共享并且图灵完备（Turing completeness），因此在对序列的非线性特征进行学习时具有一定优势。循环神经网络在自然语言处理（Natural Language Processing, NLP），例如语音识别、语言建模、机器翻译等领域有应用，也被用于各类时间序列预报。引入了卷积神经网络（Convoutional Neural Network,CNN）构筑的循环神经网络可以处理包含序列输入的计算机视觉问题。

## 最简单的RNN网络

![img](https://ai-studio-static-online.cdn.bcebos.com/383275f9f05248df9f7e6962faebaa2a637d65b49eb3458fbf29d94722557c5d)

其展开可以表示为：

![img](https://ai-studio-static-online.cdn.bcebos.com/930068b2274b46268ccff3f1357464e5d18030b64a8d41acbfd6e5d2b9b17c07)

那么数学表示的公式为：

h∗t=Whxxt+Whhht−1+bhht=σ(h∗t)o∗t=Wohht+boot=θ(o∗t)h^{t}_{*} = W_{hx}x^{t} + W_{hh}h^{t-1} + b_{h} \\ h^{t} = \sigma(h^{t}_{*}) \\ o^{t}_{*} = W_{oh} h^{t} + b_{o}\\ o^{t} = \theta (o^{t}_{*})*h*∗*t*=*W**h**x**x**t*+*W**h**h**h**t*−1+*b**h**h**t*=*σ*(*h*∗*t*)*o*∗*t*=*W**o**h**h**t*+*b**o**o**t*=*θ*(*o*∗*t*)

其中，xtx^{t}*x**t*表示t时刻的输入，oto^{t}*o**t*表示t时刻的输出，hth^{t}*h**t*表示t时刻隐藏层的状态。

由于每一步的输出不仅仅依赖当前步的网络，并且还需要前若干步网络的状态，那么这种BP改版的算法叫做Backpropagation Through Time(BPTT) , 也就是将输出端的误差值反向传递,运用梯度下降法进行更新.

## RNN的问题和改进

较为严重的是容易出现梯度消失（时间过长而造成记忆值较小）或者梯度爆炸的问题(BP算法和长时间依赖造成的)

因此, 就出现了一系列的改进的算法, 最基础的两种算法是LSTM 和 GRU.

这两种方法在面对梯度消失或者梯度爆炸的问题时，由于有特殊的方式存储”记忆”，那么以前梯度比较大的”记忆”不会像简单的RNN一样马上被抹除，因此可以一定程度上克服梯度消失问题；而针对梯度爆炸则设置阈值，超过阈值直接限制梯度。

## LSTM算法(Long Short Term Memory, 长短期记忆网络 )

LSTM(Long short-term memory,长短期记忆)是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失问题。

LSTM是有4个全连接层进行计算的，LSTM的内部结构如下图所示。 ![img](https://ai-studio-static-online.cdn.bcebos.com/25abc44d1d874e94a2f0992699b7286ae87c77a240f44e49bf12758833b25379)

其中符号含义如下： ![img](https://ai-studio-static-online.cdn.bcebos.com/05c4e513a2fe4bef90fae4df6f6b11dc0944a16390574b2095c80b34644f541e)

接下来看一下内部的具体内容： ![img](https://ai-studio-static-online.cdn.bcebos.com/3369c2a12bff48f5823490f41e53764a0da7d1ef20c6417cba2aa71e8d942153)

LSTM的核心是细胞状态——最上层的横穿整个细胞的水平线，它通过门来控制信息的增加或者删除。 STM共有三个门，分别是遗忘门，输入门和输出门。

- 遗忘门：遗忘门决定丢弃哪些信息，输入是上一个神经元细胞的计算结果ht-1以及当前的输入向量xt,二者联接并通过遗忘门后(sigmoid会决定哪些信息留下，哪些信息丢弃)，会生成一个0-1向量Γft(维度与上一个神经元细胞的输出向量Ct-1相同)，Γft与Ct-1进行点乘操作后，就会获取上一个神经元细胞经过计算后保留的信息。
- 输入门：表示要保存的信息或者待更新的信息，如上图所示是ht-1与xt的连接向量，经过sigmoid层后得到的结果Γit，这就是输入门的输出结果了。
- 输出门：输出门决定当前神经原细胞输出的隐向量ht，ht与Ct不同，ht要稍微复杂一点，它是Ct进过tanh计算后与输出门的计算结果进行点乘操作后的结果，用公式描述是：ht = tanh(ct) · Γot

## GRU（门控循环单元）

GRU是LSTM的变种，它也是一种RNN，因此是循环结构，相比LSTM而言，它的计算要简单一些，计算量也降低。

GRU 有两个有两个门，即一个重置门（reset gate）和一个更新门（update gate）。从直观上来说，重置门决定了如何将新的输入信息与前面的记忆相结合，更新门定义了前面记忆保存到当前时间步的量。如果我们将重置门设置为 1，更新门设置为 0，那么我们将再次获得标准 RNN 模型。使用门控机制学习长期依赖关系的基本思想和 LSTM 一致，但还是有一些关键区别：

- GRU 有两个门（重置门与更新门），而 LSTM 有三个门（输入门、遗忘门和输出门）。
- GRU 并不会控制并保留内部记忆（c_t），且没有 LSTM 中的输出门。
- LSTM 中的输入与遗忘门对应于 GRU 的更新门，重置门直接作用于前面的隐藏状态。 ![img](https://ai-studio-static-online.cdn.bcebos.com/6e1f4f10c72d426d8526ce30c798b3bf19e147af0ea4470d9de357937b8f1e37)
- 重置门：用来决定需要丢弃哪些上一个神经元细胞的信息，它的计算过程是将Ct-1与当前输入向量xt进行连接后，输入sigmoid层进行计算，结果为S1，再将S1与Ct-1进行点乘计算，则结果为保存的上个神经元细胞信息，用C’t-1表示。公式表示为：C’t-1 = Ct-1 · S1，S1 = sigmoid(concat(Ct-1,Xt))
- 更新门：更新门类似于LSTM的遗忘门和输入门，它决定哪些信息会丢弃，以及哪些新信息会增加。

完整公式描述为： ![img](https://ai-studio-static-online.cdn.bcebos.com/11ff4b91d2fc47e1b705615b1076628a2681cc551e1a436799311cc1b645154e)

# 数据集介绍

LCQMC（A Large-scale Chinese Question Matching Corpus）, 一个大型中文问题匹配语料库。 问题匹配是QA的一项基本任务，通常被认为是语义匹配任务，有时是释义识别任务。该任务的目标是从现有数据库中搜索与输入问题具有相似意图的问题。这里引入了一个大规模的中文问题匹配语料库（命名为 LCQMC）。LCQMC 比复述语料库更通用，因为它侧重于意图匹配而不是复述。语料库包含 260,068 个带有手动注释的问题对，分为三部分，即包含 238,766 个问题对的训练集、包含 8,802 个问题对的开发集和包含 12,500 个问题对的测试集。

| 数据集名称 | 训练集大小 | 验证集大小 | 测试集大小 |
| :--------- | :--------- | :--------- | :--------- |
| LCQMC      | 238,766    | 8,802      | 12,500     |

# 前置基础知识

## 文本语义匹配

文本语义匹配是自然语言处理中一个重要的基础问题，NLP 领域的很多任务都可以抽象为文本匹配任务。例如，信息检索可以归结为查询项和文档的匹配，问答系统可以归结为问题和候选答案的匹配，对话系统可以归结为对话和回复的匹配。语义匹配在搜索优化、推荐系统、快速检索排序、智能客服上都有广泛的应用。如何提升文本匹配的准确度，是自然语言处理领域的一个重要挑战。

- 信息检索：在信息检索领域的很多应用中，都需要根据原文本来检索与其相似的其他文本，使用场景非常普遍。
- 新闻推荐：通过用户刚刚浏览过的新闻标题，自动检索出其他的相似新闻，个性化地为用户做推荐，从而增强用户粘性，提升产品体验。
- 智能客服：用户输入一个问题后，自动为用户检索出相似的问题和答案，节约人工客服的成本，提高效率。

让我们来看一个简单的例子，比较各候选句子哪句和原句语义更相近：

原句：“车头如何放置车牌”

- 比较句1：“前牌照怎么装”
- 比较句2：“如何办理北京车牌”
- 比较句3：“后牌照怎么装”

（1）比较句1与原句，虽然句式和语序等存在较大差异，但是所表述的含义几乎相同

（2）比较句2与原句，虽然存在“如何” 、“车牌”等共现词，但是所表述的含义完全不同

（3）比较句3与原句，二者讨论的都是如何放置车牌的问题，只不过一个是前牌照，另一个是后牌照。二者间存在一定的语义相关性

所以语义相关性，句1大于句3，句3大于句2，这就是语义匹配。

## 短文本语义匹配网络

短文本语义匹配（SimilarityNet, SimNet）是一个计算短文本相似度的框架，可以根据用户输入的两个文本，计算出相似度得分。主要包括 BOW、CNN、RNN、MMDNN 等核心网络结构形式，提供语义相似度计算训练和预测框架，适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。

SimNet 模型结构如图所示，包括输入层、表示层以及匹配层。

![img](https://ai-studio-static-online.cdn.bcebos.com/953b573d100b46fe84f026c99a3a2664fbecbe680fa748eda6337548a2f4b4af)

SimilarityNet 框架

## SimilarityNet模型框架结构图

模型框架结构图如下图所示，其中 query 和 title 是数据集经过处理后的待匹配的文本，然后经过分词处理，编码成 id，经过 SimilarityNet 处理，得到输出，训练的损失函数使用的是交叉熵损失。

![img](https://ai-studio-static-online.cdn.bcebos.com/df8c743f9b3742e4a99ae6b066c133c9b119f7499e3b4addb5ce755ec3faae7e)

SimilarityNet 模型框架结构图

# LCQMC 信息检索文本相似度计算

## 环境配置

```python
# 导入必要的库
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url

print("本项目基于Paddle的版本号为："+ paddle.__version__)
```

```
# AI Studio上的PaddleNLP版本过低，所以需要首先升级PaddleNLP
!pip install paddlenlp --upgrade
```

```python
# 导入PaddleNLP相关的包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
# from utils import convert_example
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder

print("本项目基于PaddleNLP的版本号为："+ ppnlp.__version__)
```

## 加载预训练模型

```python
MODEL_NAME = "ernie-1.0"
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)
```

In [21]

```python
# 定义ERNIE模型对应的 tokenizer，并查看效果
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)
```

```python
tokens = tokenizer._tokenize("王馨语学习笔记")
print("Tokens: {}".format(tokens))

# token映射为对应token id
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))

# 拼接上预训练模型对应的特殊token ，如[CLS]、[SEP]
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
print("Tokens id: {}".format(tokens_ids))
# 转化成paddle框架数据格式
tokens_pd = paddle.to_tensor([tokens_ids])
print("Tokens : {}".format(tokens_pd))

# 此时即可输入ERNIE模型中得到相应输出
sequence_output, pooled_output = ernie_model(tokens_pd)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))
```

```python
encoded_text = tokenizer(text="王馨语学习笔记",  max_seq_len=20)
for key, value in encoded_text.items():
    print("{}:\n\t{}".format(key, value))

# 转化成paddle框架数据格式
input_ids = paddle.to_tensor([encoded_text['input_ids']])
print("input_ids : {}".format(input_ids))
segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
print("token_type_ids : {}".format(segment_ids))

# 此时即可输入 ERNIE 模型中得到相应输出
sequence_output, pooled_output = ernie_model(input_ids, segment_ids)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))
```

## 加载数据集

```python
# 首次运行需要把注释（#）去掉
!unzip -oq /home/aistudio/data/data78992/lcqmc.zip
```

```python
# 删除解压后的无用文件
!rm -r __MACOSX
```

### 查看数据

```python
import pandas as pd

train_data = "./lcqmc/train.tsv"
train_data = pd.read_csv(train_data, header=None, sep='\t')
train_data.head(10)
                    0                 1  2
0    喜欢打篮球的男生喜欢什么样的女生   爱打篮球的男生喜欢什么样的女生  1
1        我手机丢了，我想换个手机       我想买个新手机，求推荐  1
2            大家觉得她好看吗        大家觉得跑男好看吗？  0
3           求秋色之空漫画全集         求秋色之空全集漫画  1
4  晚上睡觉带着耳机听音乐有什么害处吗？      孕妇可以戴耳机听音乐吗?  0
5           学日语软件手机上的          手机学日语的软件  1
6    打印机和电脑怎样连接，该如何设置  如何把带无线的电脑连接到打印机上  0
7        侠盗飞车罪恶都市怎样改车      侠盗飞车罪恶都市怎么改车  1
8           什么花一年四季都开       什么花一年四季都是开的  1
9             看图猜一电影名            看图猜电影！  1
```

![img](https://ai-studio-static-online.cdn.bcebos.com/55b204c8f11042a0819418d34553d98bc138bd20d9584740896f6627a88fb942)

### 读取数据

```python
class lcqmcfile(DatasetBuilder):
    SPLITS = {
        'train': 'lcqmc/train.tsv',
        'dev': 'lcqmc/dev.tsv',
    }

    def _get_data(self, mode, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}

    def get_labels(self):
        return ["0", "1"]
```



```python
def load_dataset(name=None,
                 data_files=None,
                 splits=None,
                 lazy=None,
                 **kwargs):
   
    reader_cls = lcqmcfile
    print(reader_cls)
    if not name:
        reader_instance = reader_cls(lazy=lazy, **kwargs)
    else:
        reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

    datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
    return datasets
```



```python
train_ds, dev_ds = load_dataset(splits=["train", "dev"])
<class '__main__.lcqmcfile'>
```

## 模型构建

```python
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader

batch_size = 64
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),       # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")                               # label
): [data for data in fn(samples)]
```



```python
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 训练配置

```python
from paddlenlp.transformers import LinearDecayWithWarmup

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()
```

## 模型训练

```python
import paddle.nn.functional as F
from utils import evaluate

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)
```

用时约45分钟。 

## 预测模型

### 测试结果

```python
from utils import predict
import pandas as pd

label_map = {0:'0', 1:'1'}

def preprocess_prediction_data(data):
    examples = []
    for query, title in data:
        examples.append({"query": query, "title": title})
        # print(len(examples),': ',query,"---", title)
    return examples
test_file = 'lcqmc/test.tsv'
data = pd.read_csv(test_file, sep='\t')
# print(data.shape)
data1 = list(data.values)
examples = preprocess_prediction_data(data1)
```

### 输出 tsv 文件

In [ ]

```python
results = predict(
        model, examples, tokenizer, label_map, batch_size=batch_size)

for idx, text in enumerate(examples):
    print('Data: {} \t Label: {}'.format(text, results[idx]))

data2 = []
for i in range(len(data1)):
    data2.extend(results[i])

data['label'] = data2
print(data.shape)
data.to_csv('lcqmc.tsv',sep='\t')
```

Data: {'query': '英雄联盟什么英雄最好', 'title': '英雄联盟最好英雄是什么'} 	 Label: 1
Data: {'query': '这是什么意思，被蹭网吗', 'title': '我也是醉了，这是什么意思'} 	 Label: 0
Data: {'query': '现在有什么动画片好看呢？', 'title': '现在有什么好看的动画片吗？'} 	 Label: 1
Data: {'query': '请问晶达电子厂现在的工资待遇怎么样要求有哪些', 'title': '三星电子厂工资待遇怎么样啊'} 	 Label: 0
Data: {'query': '文章真的爱姚笛吗', 'title': '姚笛真的被文章干了吗'} 	 Label: 0
Data: {'query': '送自己做的闺蜜什么生日礼物好', 'title': '送闺蜜什么生日礼物好'} 	 Label: 1
Data: {'query': '近期上映的电影', 'title': '近期上映的电影有哪些'} 	 Label: 1
Data: {'query': '求英雄联盟大神带？', 'title': '英雄联盟，求大神带~'} 	 Label: 1
Data: {'query': '如加上什么部首', 'title': '给东加上部首是什么字？'} 	 Label: 0
Data: {'query': '杭州哪里好玩', 'title': '杭州哪里好玩点'} 	 Label: 1
Data: {'query': '这是什么乌龟值钱吗', 'title': '这是什么乌龟！值钱嘛？'} 	 Label: 1
Data: {'query': '心各有所属是什么意思？', 'title': '心有所属是什么意思?'} 	 Label: 0
Data: {'query': '什么东西越热爬得越高', 'title': '什么东西越热爬得很高'} 	 Label: 1
Data: {'query': '世界杯哪位球员进球最多', 'title': '世界杯单界进球最多是哪位球员'} 	 Label: 0
Data: {'query': '韭菜多吃什么好处', 'title': '多吃韭菜有什么好处'} 	 Label: 1
Data: {'query': '云赚钱怎么样', 'title': '怎么才能赚钱'} 	 Label: 0
Data: {'query': '何炅结婚了嘛', 'title': '何炅结婚了么'} 	 Label: 1
Data: {'query': '长的清新是什么意思', 'title': '小清新的意思是什么'} 	 Label: 0
Data: {'query': '我们可以结婚了吗？', 'title': '在熙结婚了吗？'} 	 Label: 0
Data: {'query': '想买男人酒补肾壮阳酒哪里有啊', 'title': '哪里有男人酒补肾壮阳酒'} 	 Label: 1
Data: {'query': '淘宝上怎么用信用卡分期付款', 'title': '淘宝怎么分期付款，没有信用卡'} 	 Label: 0
Data: {'query': '最近有没有什么好看的韩剧', 'title': '最近有什么好看的韩剧'} 	 Label: 1
Data: {'query': '《校花的贴身高手》中的林逸', 'title': '校花贴身高手'} 	 Label: 1
Data: {'query': '叔叔是什么人', 'title': '我是叔叔的什么人'} 	 Label: 0
Data: {'query': '这姑娘漂亮不', 'title': '我姑娘漂亮吧'} 	 Label: 1
