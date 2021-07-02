# GPT
## 1. 介绍

2018 年 6 月，OpenAI 发表论文介绍了自己的语言模型 GPT，GPT 是“Generative Pre-Training”的简称，它基于 Transformer 架构，GPT模型先在大规模语料上进行无监督预训练、再在小得多的有监督数据集上为具体任务进行精细调节（fine-tune）的方式。先训练一个通用模型，然后再在各个任务上调节，这种不依赖针对单独任务的模型设计技巧能够一次性在多个任务中取得很好的表现。这中模式也是 2018 年中自然语言处理领域的研究趋势，就像计算机视觉领域流行 ImageNet 预训练模型一样。


### 1.1 GPT的动机

NLP 领域中只有小部分标注过的数据，而有大量的数据是未标注，如何只使用标注数据将会大大影响深度学习的性能，所以为了充分利用大量未标注的原始文本数据，需要利用无监督学习来从文本中提取特征，最经典的例子莫过于词嵌入技术。但是词嵌入只能 word-level 级别的任务（同义词等），没法解决句子、句对级别的任务（翻译、推理等）。出现这种问题原因有两个：

+ 不清楚下游任务，所以也就没法针对性的进行优化；
+ 就算知道了下游任务，如果每次都要大改模型也会得不偿失。

为了解决以上问题，作者提出了 GPT 框架，用一种半监督学习的方法来完成语言理解任务，GPT 的训练过程分为两个阶段：无监督Pre-training 和 有监督Fine-tuning。在Pre-training阶段使用单向 Transformer 学习一个语言模型，对句子进行无监督的 Embedding，在fine-tuning阶段，根据具体任务对 Transformer 的参数进行微调，目的是在于学习一种通用的 Representation 方法，针对不同种类的任务只需略作修改便能适应。


## 2. 模型结构


GPT 使用 Transformer 的 Decoder 结构，并对 Transformer Decoder 进行了一些改动，原本的 Decoder 包含了两个 Multi-Head Attention 结构，GPT 只保留了 Mask Multi-Head Attention，如下图所示。

![](https://raw.githubusercontent.com/w5688414/paddleImage/main/bert_family_img/mask_multi_head_attention.jpeg)

GPT 使用句子序列预测下一个单词，因此要采用 Mask Multi-Head Attention 对单词的下文遮挡，防止信息泄露。例如给定一个句子包含4个单词 [A, B, C, D]，GPT 需要利用 A 预测 B，利用 [A, B] 预测 C，利用 [A, B, C] 预测 D。如果利用A 预测B的时候，需要将 [B, C, D] Mask 起来。

Mask 操作是在 Self-Attention 进行 Softmax 之前进行的，具体做法是将要 Mask 的位置用一个无穷小的数替换 -inf，然后再 Softmax，如下图所示。

![](https://raw.githubusercontent.com/w5688414/paddleImage/main/bert_family_img/softmax_mask.jpeg)

<center>Softmax 之前需要 Mask</center>

![gpt softmax](https://raw.githubusercontent.com/w5688414/paddleImage/main/bert_family_img/gpt_softmax.jpeg)
<center>GPT Softmax</center>


可以看到，经过 Mask 和 Softmax 之后，当 GPT 根据单词 A 预测单词 B 时，只能使用单词 A 的信息，根据 [A, B] 预测单词 C 时只能使用单词 A, B 的信息。这样就可以防止信息泄露。

下图是 GPT 整体模型图，其中包含了 12 个 Decoder。


![](https://raw.githubusercontent.com/w5688414/paddleImage/main/bert_family_img/gpt_model.jpeg)

GPT只使用了 Transformer 的 Decoder 部分，并且每个子层只有一个 Masked Multi Self-Attention（768 维向量和 12 个 Attention Head）和一个 Feed Forward，共叠加使用了 12 层的 Decoder。

这里简单解释下为什么只用 Decoder 部分：语言模型是利用上文预测下一个单词的，因为 Decoder 使用了 Masked Multi Self-Attention 屏蔽了单词的后面内容，所以 Decoder 是现成的语言模型。又因为没有使用 Encoder，所以也就不需要 encoder-decoder attention 了。

## 3. GPT训练过程

### 3.1 无监督的预训练


无监督的预训练（Pretraining），具体来说，给定一个未标注的预料库$U=\{u_{1},u_{2},...,u_{n}\}$，我们训练一个语言模型，对参数进行最大（对数）似然估计：

$$L_{1}(U)=\sum_{i}log P(u_{i}|u_{1},...,u_{k-1};\Theta)$$

其中，k 是上下文窗口的大小，P 为条件概率，$\Theta$为条件概率的参数，参数更新采用随机梯度下降（GPT实验实现部分具体用的是Adam优化器，并不是原始的随机梯度下降，Adam 优化器的学习率使用了退火策略）。

训练的过程也非常简单，就是将 n 个词的词嵌入$W_{e}$加上位置嵌入$W_{p}$，然后输入到 Transformer 中，n 个输出分别预测该位置的下一个词

可以看到 GPT 是一个单向的模型，GPT 的输入用 $h_{0}$ 表示，0代表的是输入层，$h_{0}$的计算公式如下

$$h_{0}=UW_{e}+W_{p}$$

$W_{e}$是token的Embedding矩阵，$W_{p}$是位置编码的 Embedding 矩阵。用 voc 表示词汇表大小，pos 表示最长的句子长度，dim 表示 Embedding 维度，则$W_{p}$是一个 pos×dim 的矩阵，$W_{e}$是一个 voc×dim 的矩阵。在GPT中，作者对position embedding矩阵进行随机初始化，并让模型自己学习，而不是采用正弦余弦函数进行计算。

得到输入 $h_{0}$ 之后，需要将 $h_{0}$ 依次传入 GPT 的所有 Transformer Decoder 里，最终得到$h_{n}$。

$$h_{l}=transformer\_block(h_{l-1}), \forall l \in [1,n]$$

n 为神经网络的层数。最后得到$h_{n}$再预测下个单词的概率。

$$P(u)=softmax(h_{n}W_{e}^T)$$


### 3.2 有监督的Fine-Tuning

预训练之后，我们还需要针对特定任务进行 Fine-Tuning。假设监督数据集合$C$的输入$X$是一个序列$x^1,x^2,...,x^m$，输出是一个分类y的标签 ，比如情感分类任务

我们把$x^1,..,x^m$输入 Transformer 模型，得到最上层最后一个时刻的输出$h_{l}^m$，将其通过我们新增的一个 Softmax 层（参数为$W_{y}$）进行分类，最后用交叉熵计算损失，从而根据标准数据调整 Transformer 的参数以及 Softmax 的参数 $W_{y}$。这等价于最大似然估计：


$$P(y|x^1,...,x^m)=softmax(h_{l}^mW_{y})$$

$W_{y}$表示预测输出时的参数，微调时候需要最大化以下函数:

$$L_{2}(C)=\sum_{x,y}log P(y|x^1,..,x^m)$$

正常来说，我们应该调整参数使得$L_{2}$最大，但是为了提高训练速度和模型的泛化能力，我们使用 Multi-Task Learning，GPT 在微调的时候也考虑预训练的损失函数，同时让它最大似然$L_{1}$和$L_{2}$

$$L_{3}(C)=L_{2}(C)+\lambda \times L_{1}(C) $$ 

这里使用的$L_{1}$还是之前语言模型的损失（似然），但是使用的数据不是前面无监督的数据$U$，而是使用当前任务的数据$C$，而且只使用其中的$X$，而不需要标签y。

### 3.3 其它任务

针对不同任务，需要简单修改下输入数据的格式，例如对于相似度计算或问答，输入是两个序列，为了能够使用 GPT，我们需要一些特殊的技巧把两个输入序列变成一个输入序列

![](https://raw.githubusercontent.com/w5688414/paddleImage/main/bert_family_img/gpt_task.png)

+ Classification：对于分类问题，不需要做什么修改
+ Entailment：对于推理问题，可以将先验与假设使用一个分隔符分开
+ Similarity：对于相似度问题，由于模型是单向的，但相似度与顺序无关，所以要将两个句子顺序颠倒后，把两次输入的结果相加来做最后的推测
+ Multiple-Choice：对于问答问题，则是将上下文、问题放在一起与答案分隔开，然后进行预测 


## 4. GPT特点
### 优点

+ 特征抽取器使用了强大的 Transformer，能够捕捉到更长的记忆信息，且较传统的 RNN 更易于并行化；
+ 方便的两阶段式模型，先预训练一个通用的模型，然后在各个子任务上进行微调，减少了传统方法需要针对各个任务定制设计模型的麻烦。

### 缺点
+ GPT 最大的问题就是传统的语言模型是单向的；我们根据之前的历史来预测当前词。但是我们不能利用后面的信息。比如句子 The animal didn’t cross the street because it was too tired。我们在编码 it 的语义的时候需要同时利用前后的信息，因为在这个句子中，it 可能指代 animal 也可能指代 street。根据 tired，我们推断它指代的是 animal。但是如果把 tired 改成 wide，那么 it 就是指代 street 了。Transformer 的 Self-Attention 理论上是可以同时关注到这两个词的，但是根据前面的介绍，为了使用 Transformer 学习语言模型，必须用 Mask 来让它看不到未来的信息，所以它也不能解决这个问题。


## 5. GPT 与 ELMo的区别

GPT 与 ELMo 有两个主要的区别：

1. 模型架构不同：ELMo 是浅层的双向 RNN；GPT 是多层的 Transformer encoder

2. 针对下游任务的处理不同：ELMo 将词嵌入添加到特定任务中，作为附加功能；GPT 则针对所有任务微调相同的基本模型

## 参考文献
[Improving Language Understanding by Generative Pre-Training](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)



