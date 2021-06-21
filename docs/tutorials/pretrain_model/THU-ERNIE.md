# ERNIE：Enhanced Language Representation with Informative Entities
<br>

## 1. THU-ERNIE的由来

当前的预训练模型（比如BERT、GPT等）往往在大规模的语料上进行预训练，学习丰富的语言知识，然后在下游的特定任务上进行微调。但这些模型基本都没有使用**知识图谱（KG）**这种结构化的知识，而KG本身能提供大量准确的知识信息，通过向预训练语言模型中引入这些外部知识可以帮助模型理解语言知识。基于这样的考虑，作者提出了一种融合知识图谱的语言模型**ERNIE**，由于该模型是由清华大学提供的，为区别百度的ERNIE，故本文后续将此模型标记为[**THU-ERNIE**](https://arxiv.org/pdf/1905.07129.pdf)。

这个想法很好，但将知识图谱的知识引入到语言模型存在**两个挑战**：

* Structured Knowledge Encoding：如何为预训练模型提取和编码知识图谱的信息？
* Heterogeneous Information Fusion：语言模型和知识图谱对单词的表示（representation）是完全不同的两个向量空间，这种情况下如何将两者进行融合？

对于第一个问题，**THU-ERNIE**使用[TAGME](https://arxiv.org/pdf/1006.3498v1.pdf)提取文本中的实体，并将这些实体链指到KG中的对应实体对象，然后找出这些实体对象对应的embedding，这些embedding是由一些知识表示方法，例如[TransE](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)训练得到的。

对于第二个问题，**THU-ERNIE**在BERT模型的基础上进行改进，除了MLM、NSP任务外，重新添加了一个和KG相关的预训练目标：Mask掉token和entity (实体) 的对齐关系，并要求模型从图谱的实体中选择合适的entity完成这个对齐。

## 2. THU-ERNIE的模型结构

![image-20210616194846237](https://raw.githubusercontent.com/1649759610/images_for_blog/master/image-20210616194846237.png)

<center>图1 THU-ERNIE的模型架构</center>

**THU-ERNIE**在预训练阶段就开始了与KG的融合，如**图1a**所示，**THU-ERNIE**是由两种类型的Encoder堆叠而成：**T-Encoder**和**K-Encoder**。其中**T-Encoder**在下边堆叠了$N$层，**K-Encoder**在上边堆叠了$M$层，所以整个模型共有$N+M$层，**T-Encoder**的输出和相应的KG实体知识作为**K-Encoder**的输入。

从功能上来讲，**T-Encoder**负责从输入序列中捕获词法和句法信息；**K-Encoder**负责将KG知识和从**T-Encoder**中提取的文本信息进行融合，其中KG知识在这里主要是实体，这些实体是通过TransE模型训练出来的。

**THU-ERNIE**中的**T-Encoder**的结构和BERT结构是一致的，**K-Encoder**则做了一些改变，**K-Encoder**对**T-Encoder**的输出序列和实体输入序列分别进行Multi-Head Self-Attention操作，之后将两者通过Fusion层进行融合。

## 3. K-Encoder融合文本信息和KG知识

本节将详细探讨**K-Encoder**的内部结构以及**K-Encoder**是如何融合预训练文本信息和KG知识的。**图1b**展示了**K-Encoder**的内部细节信息。

我们可以看到，其对文本序列 (token Input) 和KG知识(Entity Input)分别进行Multi-Head Self-Attention(MH-ATT)操作，假设在第$i$层中，token Input对应的embedding是$\{w_{1}^{(i-1)},w_{2}^{(i-1)},...,w_{n}^{(i-1)}\}$，Entity Input对应的embedding是$\{ e_1^{(i-1)},e_2^{(i-1)},...,e_n^{(i-1)}\}$，则Multi-Head Self-Attention操作的公式可以表示为：

$$
\{\tilde{w}_{1}^{(i-1)},\tilde{w}_{2}^{(i-1)},...,\tilde{w}_{n}^{(i-1)}\} = \text{MH-ATT}(\{w_{1}^{(i-1)},w_{2}^{(i-1)},...,w_{n}^{(i-1)}\}) \\
\{\tilde{e}_{1}^{(i-1)},\tilde{e}_{2}^{(i-1)},...,\tilde{e}_{m}^{(i-1)}\} = \text{MH-ATT}(\{e_{1}^{(i-1)},e_{2}^{(i-1)},...,e_{m}^{(i-1)}\}) 
$$

然后Entity序列的输出将被对齐到token序列的第一个token上，例如实体"bob dylan"将被对齐到第一个单词"bob"上。接下里将这些MH-ATT的输入到Fusion层，在这里将进行文本信息和KG知识的信息融合。因为有些token没有对应的entity，有些token有对应的entity，所以这里需要分两种情况讨论。

对于那些**有**对应entity的token，信息融合的过程是这样的：

$$
h_j = \sigma(\tilde{W}_t^{(i)}\tilde{w}_j^{(i)}+\tilde{W}_e^{(i)}\tilde{e}_k^{(i)}+\tilde{b}^{(i)}) \\
w_j^{(i)} = \sigma({W}_t^{(i)}{h}_j+b_t^{(i)}) \\ 
e_k^{(i)} = \sigma({W}_e^{(i)}{h}_j+b_e^{(i)})
$$


对于那些**没有**对应entity的token，信息融合的过程是这样的：

$$
h_j = \sigma(\tilde{W}_t^{(i)}\tilde{w}_j^{(i)}+\tilde{b}^{(i)}) \\
w_j^{(i)} = \sigma({W}_t^{(i)}{h}_j+b_t^{(i)}) 
$$

其中这里的$\sigma(\cdot)$是个非线性的激活函数，通常可以使用GELU函数。最后一层的输出将被视作融合文本信息和KG知识的最终向量。

## 4. THU-ERNIE的预训练任务

在预训练阶段，**THU-ERNIE**的预训练任务包含3个任务：MLM、NSP和dEA。dEA将随机地Mask掉一些token-entity对，然后要求模型在这些对齐的token上去预测相应的实体分布，其有助于将实体注入到**THU-ERNIE**模型的语言表示中。

由于KG中实体的数量往往过于庞大，对于要进行这个任务的token来讲，**THU-ERNIE**将会给定小范围的实体，让该token在这个范围内去计算要输出的实体分布，而不是全部的KG实体。

给定token序列$\{w_{1},w_{2},...,w_{n}\}$和对应的实体序$\{ e_1,e_2,...,e_m\}$，对于要对齐的token $w_i$来讲，相应的对齐公式为：

$$
p(e_j|w_i) = \frac{exp(\text{linear}(w_i^o) \cdot e_j)}{\sum_{k=1}^{m}exp(\text{linear}(w_i^{o}) \cdot e_k)}
$$

类似与BERT对token的Mask策略，**THU-ERNIE**在Mask token-entity对齐的时候也采用的一定的策略，如下：

1. 以5%的概率去随机地替换实体，让模型去预测正确的entity。
2. 以15%的概率直接Mask掉token-entity，让模型去预测相应的entity。
3. 以80%的概率保持token-entity的对齐不变，以让模型学到KG知识，提升语言理解能力。

最终，**THU-ERNIE**的总的预训练损失是由MLM、NSP和dEA三者的加和。

## 6. 参考资料

1. [ERNIE：Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf)
2. [ERNIE Githut](https://github.com/thunlp/ERNIE)

