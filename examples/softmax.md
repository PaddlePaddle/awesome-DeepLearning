一.深度学习基础题

##**softmax**

首先，我们来想一下，凭什么可以训练出[《词向量概念扫盲》](http://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ%3D%3D&chksm=970c2963a07ba075f3877dc9cbecfbdd96ba6e31e661de64f52504c75b60782d9c5bf53d0b0b&idx=1&mid=2247483829&scene=21&sn=e5cddcdaf83d5d97515972bda54c496b#wechat_redirect)中描述的这么棒的可以编码语义的词向量呢？其实，只要是用无监督的方法去训练词向量，那么这个模型一定是基于**“词共现”****(word co-occurrence)**信息来实现的。

 

设想一下，“萌”的语义跟什么最相近呢？有人会想到“可爱”，有人会想到“妹子”，因为很大程度上，这些词会同时出现在某段文本的中，而且往往距离很近！比如“萌妹子”、“这个妹子好可爱”。正是因为这种词共现可以反映语义，所以我们就可以基于这种现象来训练出词向量。

 

既然语义相近的两个词（即差别很小的两个词向量）在文本中也会趋向于挨得近，那如果我们可以找到一个模型，它可以在给定一个词向量时，计算出这个词附近出现每个词的概率（即一个词就看成一个类别，词典中有多少词，就是多少个类别，计算出给定输入下，每个类别的概率），那么训练这个模型不就把问题解决了。

 

我们就将词典大小设为D，用![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFNNDJkY253Ykc2V1h3TE0zT3FKTzJqcXlWTjQyV2dpY1pPN0Q2UzVpYlhYaWJFaWNpYnFINXl3cU1hdy8w?x-oss-process=image/format,png)、![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFrWnZPUTlRclhYekhmaWNabVY2Y2VxbGF1bEtHMHQ3TlIyaFJxTEU1N1NLUkJPeWpaMW44Z0hBLzA?x-oss-process=image/format,png)、...![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEF2MmZJNHl3RFVzWGQ4Q21BMlFJdjc3OFhpY3dxM1dPUWxDSmtkQ2lhVVBDa0IyMDYwazA1eFY4dy8w?x-oss-process=image/format,png)表示词典中的每个词。

 

如下图：

<img src="https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFtMDBHSFpnV3RCSmRpYndydFVPQU9aVDloQ0FJQWRwUGxwNm5CY0lCaWNlRHIyM1d3c1RJSnA1dy8w?x-oss-process=image/format,png" alt="img" style="zoom:50%;" />

这就是简单的softmax分类器，所以这个model的假设函数就是简单的：

![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEE3dnRpYXFWVHhIM09CdE9xekljYndpYWljaFJUQTJFdlpLbEpoWHlNT3hvU3l2Q2x1VzJibGJYQ2cvMA?x-oss-process=image/format,png)

从这个model中也能看出，模型的输入不仅是输入，而且是其他输入的参数！所以这个model的参数是维度为 **D\*embed_dim 的矩阵**（每行就是一个用户定义的embed_dim大小的词向量，词典中有D个词，所以一共有D行），而且输入也是从这个矩阵中取出的某一行）。

 

假设函数有了，那么根据[《一般化机器学习》](http://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ%3D%3D&chksm=970c2ba8a07ba2be6d03728cab594fb464276cc93f9ab03b7c6441490b49c933bd431a33d119&idx=1&mid=2247484286&scene=21&sn=68c1b5bf8bd6530ed6c779efbad67404#wechat_redirect)，我们需要定义**损失函数**。当然，根据前面所说的词共现信息来定义。

 

为了好表示，我们将模型输入的词称为**中心词(central word)**，记为![w_c](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEEzb3lYOEJYbjFVTE9IVWNnQnhCQTNuWE9PbUtpYnhhdUtra212bFBGa0g3S2tHY0xpYzJvRVlGZy8w?x-oss-process=image/format,png)，将这个词两边的词记为**目标词(objected word)**，记为![w_o](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEF1Y0Qwa2tpY2NNbFdIYnJsSjZFQWlhZVk1TGo4S3JMNnJiN2tVVDhpY2ptT0VobUhsc0xzeGJ3WncvMA?x-oss-process=image/format,png)，假如我们只将中心词附近的m个词认为是它的共现词（也就是中心词左边的m个词以及中心词右边的m个词），那么目标词一共有2m个，分别记为![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFnR1pUVTExaWFPaWIzaWFGYWFiZnowak1yZkNxRTBYQXZKMlo3aWJBMVlYMXI1RlZyWmp5aWJERVhHQS8w?x-oss-process=image/format,png)、![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFpYW5IeWZ3c3BrRkl0aWJnZzl3Ym52NmljaHhON1VEblZrdm9UdFB4aWJYc2s2ZnhjVEpLc205Y3ZRLzA?x-oss-process=image/format,png)、...![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEFLcDdWa09HUEo5aWFGSG1wNU8xckNMSWlhZ3VwRTl4TWJqN1lZYVQ1SGhQdjRmeFczWkt1UHVCdy8w?x-oss-process=image/format,png)。（下文将会看到，在整个句子的视角下，m被称为**窗口大小**）

 

如果我们令m=1，那么对于下面这个**长度为****T=10**句子：

 

今天 我 看见 一只 可爱的 猫 坐 在 桌子 上。

 

那么当我们将“猫”看作中心词时，目标词就是“可爱的”和“坐”，即

今天 我 看见 一只 【可爱的 猫 坐】 在 桌子 上。

我们就认为这两个词跟猫的语义是相关的，其他词跟猫是否相关我们不清楚。**所以我们要争取让P(可爱的|猫)、 P(坐|猫)尽可能的大**。

 

讲到这里，最容易想到的就是使用似然函数了。由于这里类别特别多，所以算出来的每个概率都可能非常小，为了避免浮点下溢（值太小，容易在计算机中被当成0，而且容易被存储浮点数的噪声淹没），更明智的选择是使用对数似然函数。所以对于一段长度为T的训练文本，损失函数即：

![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEF4YVd0aWNnWldmMUNrRDd1bnUxNUtDc1lERzZlaWFUVEtCMzJZRGRmVTdGT1NyTE9yTGtrSWtqZy8w?x-oss-process=image/format,png)

这里要让长度为m的窗口滑过训练文本中的每个词，滑到每个词时，都要计算2m次后验概率。而每次计算后验概率都要用到softmax函数，而回顾一下softmax函数，它的分母是很恐怖的：

![img](https://imgconvert.csdnimg.cn/aHR0cDovL21tYml6LnFwaWMuY24vbW1iaXpfcG5nLzVma25iNDFpYjlxRW9taFU2YWFEMDhjdzFZaWE2ZzBWUEE3dnRpYXFWVHhIM09CdE9xekljYndpYWljaFJUQTJFdlpLbEpoWHlNT3hvU3l2Q2x1VzJibGJYQ2cvMA?x-oss-process=image/format,png)

类别越多，分母越长。而我们这里类别数等于词典大小！所以词典有10万个单词的话，分母要计算10万次指数函数？所以直接拿最优化算法去优化这个损失函数的话，训练难度会很大。

 

一种很巧妙的方法是将原来计算复杂度为D的分母（要计算D次指数函数）通过构造一棵“**胡夫曼二叉树****(Huffman binary tree)**”来将原来扁平的“softmax”给变成树状的softmax，从而将softmax的分母给优化成计算复杂度为log D。这种树形的softmax也叫**分层softmax（\**Hierarchical Softmax\**）**。

 

还有一种优化方法是**负采样（Negative Sampling）**，这种方法可以近似计算softmax的对数概率。对使用分层softmax和负采样优化模型计算复杂度感兴趣的同学，可以看下面这篇论文：

 

Mikolov,T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013, October 17).Distributed Representations of Words and Phrases and their Compositionality.arXiv.org

这个看起来这么简洁优美的model就是Mikolov在2013年提出来的**Skip-gram（简称SG）**，这也是大名鼎鼎的开源词向量工具**word2vec**背后的主力model之一（另一个模型是连续词袋模型，即**cBoW**）。