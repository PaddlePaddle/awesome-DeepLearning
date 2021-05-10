# ERINE

## 1. ERINE是什么
[ERINE](https://arxiv.org/pdf/1904.09223.pdf)是百度发布一个预训练模型，它通过引入三种级别的Knowledge Masking帮助模型学习语言知识，在多项任务上超越了BERT。在模型结构方面，它采用了Transformer的Encoder部分作为模型主干进行训练，如 **图1** (图片来自网络)所示。

<br></br>
<center><img src="https://raw.githubusercontent.com/1649759610/images_for_blog/master/35c0a1190395471d9e4b6a2e536d34efb4dcd463cef443ad8bcaaf90a943d615.png" width=50%></center>
<center><br>图1 Transformer的Encoder部分</br></center>
<br></br>

关于ERNIE网络结构(Transformer Encoder)的工作原理，这里不再展开讨论。接下来，我们将聚焦在ERNIE本身的主要改进点进行讨论，即三个层级的Knowledge Masking 策略。这三种策略都是应用在ERNIE预训练过程中的预训练任务，期望通过这三种级别的任务帮助ERNIE学到更多的语言知识。

## 2. Knowledge Masking Task
训练语料中蕴含着大量的语言知识，例如词法，句法，语义信息，如何让模型有效地学习这些复杂的语言知识是一件有挑战的事情。BERT使用了MLM（masked language-model）和NSP（Next Sentence Prediction）两个预训练任务来进行训练，这两个任务可能并不能让BERT学到那么多复杂的语言知识，特别是后来多个研究人士提到NSP任务是比较简单的任务，它实际的作用不是很大。

----
**说明：**

masked language-model（MLM）是指在训练的时候随即从输入预料上mask掉一些单词，然后通过的上下文预测该单词，该任务非常像我们在中学时期经常做的完形填空。
Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。

----

考虑到这一点，ERNIE提出了Knowledge Masking的策略，其包含三个级别：ERNIE将Knowledge分成了三个类别：`token级别(Basic-Level)`、`短语级别(Phrase-Level)` 和 `实体级别(Entity-Level)`。通过对这三个级别的对象进行Masking，提高模型对字词、短语的知识理解。

> token级别：在英文中一个token就是一个单词，在中文中一个token就是一个字

**图2**展示了这三个级别的Masking策略和BERT Masking的对比，显然，Basic-Level Masking 同BERT的Masking一样，随机地对某些单词(如 written)进行Masking，在预训练过程中，让模型去预测这些被Mask后的单词；Phrase-Level Masking 是对语句中的短语进行masking，如 a series of；Entity-Level Masking是对语句中的实体词进行Masking，如人名 J. K. Rowling。

<br></br>
<center><img src="https://raw.githubusercontent.com/1649759610/images_for_blog/master/093f3ff3205d43bf9521179e3db78dc1f427474122ac4dc59ab0dc1263396f75.png" width=70%></center>
<center><br>图2 ERNIE和BERT的Masking策略对比</br></center>
<br></br>

除了上边的Knowledge Masking外，ERNIE还采用多个**异源语料**帮助模型训练，例如对话数据，新闻数据，百科数据等等。通过这些改进以保证模型在字词、语句和语义方面更深入地学习到语言知识。当ERINE通过这些预训练任务学习之后，就会变成一个更懂语言知识的预训练模型，接下来，就可以应用ERINE在不同的**下游任务**进行微调，提高下游任务的效果。例如，文本分类任务。

> **异源语料** ：来自不同源头的数据，比如百度贴吧，百度新闻，维基百科等等

