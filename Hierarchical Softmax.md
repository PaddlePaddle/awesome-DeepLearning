# word2vector——Hierarchical Softmax

## 背景

在统计自然语言处理中，语言模型指的是计算**一个句子的概率模型**。

传统的语言模型中词的表示是原始的、面向字符串的。两个语义相似的词的字符串可能完全不同，比如“番茄”和“西红柿”。这给所有NLP任务都带来了挑战——**字符串本身无法储存语义信息**。该挑战突出表现在模型的平滑问题上：标注语料是有限的，而语言整体是无限的，传统模型无法借力未标注的海量语料，只能靠人工设计平滑算法，而这些算法往往效果甚微。

神经概率语言模型（Neural Probabilistic Language Model）中词的表示是向量形式、面向语义的。两个语义相似的词对应的向量也是相似的，具体反映在夹角或距离上。甚至一些语义相似的二元词组中的词语对应的向量做线性减法之后得到的向量依然是相似的。**词的向量表示可以显著提高传统NLP任务的性能**，例如《基于神经网络的高性能依存句法分析器》中介绍的词、词性、依存关系的向量化对正确率的提升等。

从向量的角度来看，字符串形式的词语其实是更高维、更稀疏的向量。若词汇表大小为N，每个字符串形式的词语字典序为i，则其被表示为一个N维向量，该向量的第i维为1，其他维都为0。汉语的词汇量大约在十万这个量级，十万维的向量对计算来讲绝对是个维度灾难。而word2vec得到的词的向量形式（下文简称“词向量”，更学术化的翻译是“词嵌入”）则可以自由控制维度，一般是100左右。

word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习某个语言模型而产生的中间结果。具体来说，“某个语言模型”指的是“CBOW”和“Skip-gram”。具体学习过程会用到两个降低复杂度的近似方法——Hierarchical Softmax或Negative Sampling。两个模型乘以两种方法，一共有四种实现。

（CBOW、Skip-gram、Negative Sampling详见https://aistudio.baidu.com/aistudio/projectdetail/2198230
，此处主要介绍Hierarchical Softmax）

### 提出背景



### 理论基础

#### Huffman Tree

给定N个权值作为N个叶子结点，构造一棵二叉树，若该树的带权路径长度达到最小，称这样的二叉树为最优二叉树，也称为哈夫曼树(Huffman Tree)。哈夫曼树是带权路径长度最短的树，权值较大的结点离根较近。

	构造霍夫曼树步骤：

	假设有n个权值，则构造出的哈夫曼树有n个叶子结点。 n个权值分别设为 w1、w2、…、wn，则哈夫曼树的构造规则为：

	(1) 将w1、w2、…，wn看成是有n 棵树的森林(每棵树仅有一个结点)；

	(2) 在森林中选出两个根结点的权值最小的树合并，作为一棵新树的左、右子树，且新树的根结点权值为其左、右子树根结点权值之和；

	(3)从森林中删除选取的两棵树，并将新树加入森林；

	(4)重复(2)、(3)步，直到森林中只剩一棵树为止，该树即为所求得的哈夫曼树。
    
 构造实例如下：

![](https://ai-studio-static-online.cdn.bcebos.com/f3d9683579ad451db9adc85613b9100414c88a50160048a08b5cc276047ecf79)


#### Logistic Regression

logistic回归是一种广义线性回归（generalized linear model），因此与多重线性回归分析有很多相同之处。它们的模型形式基本上相同，都具有 w‘x+b，其中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将w‘x+b作为因变量，即y =w‘x+b，而**logistic回归则通过函数L将w‘x+b对应一个隐状态p，p =L(w‘x+b),然后根据p 与1-p的大小决定因变量的值**。如果L是logistic函数，就是logistic回归，如果L是多项式函数就是多项式回归。

Softmax其实就是多分类的Logistic Regression，相当于把很多个Logistic Regression组合在一起。

Logistic Regression在这里的应用就是判断在哈夫曼树中走左子树还是右子树，其输出的值就是走某一条的概率。

![](https://ai-studio-static-online.cdn.bcebos.com/f032f0be4f4f460b88d1bc689048a37cdbffb9db5abf45918e2464c66fc95301)


##  详细介绍（在CBOW中的层次Softmax）

对于神经网络模型多分类，最朴素的做法是softmax回归，softmax回归需要对语料库中每个词语（类）都计算一遍输出概率并进行归一化，在几十万词汇量的语料上无疑是令人头疼的。

在这样的情况下，hierarchical softmax被提出用以改进运算，其主要优点是：不需要评估神经网络中的W个输出节点来获得概率分布，只需要评估大约log2(W)个节点。

CBOW是已知上下文，估算当前词语的语言模型。其学习目标是最大化对数似然函数：

![](https://ai-studio-static-online.cdn.bcebos.com/747426c2bca9442b8ff74dbe59de39c872158d22418243ada16026c0d3974484)

	其中，w表示语料库C中任意一个词。从上图可以看出，对于CBOW：

	输入层是上下文的词语的词向量。

	投影层对其求和，所谓求和，就是简单的向量加法。

	输出层输出最可能的w。由于语料库中词汇量是固定的|C|个，所以上述过程其实可以看做一个多分类问题。给定特征，从|C|个分类中挑一个。

![](https://ai-studio-static-online.cdn.bcebos.com/1f5604dfc1a447abbc93e5ad15c8a96071c75ed5019443b990134a765e26f53a)

每个叶子节点代表语料库中的一个词，于是每个词语都可以被01唯一的编码，并且其编码序列对应一个事件序列，于是我们可以计算条件概率 

	每个单词w都可以通过从树根开始的适当路径到达。设n(w，j)是从根到w的路径上的第j个节点，设L(w)是这条路径的长度，那么n(w，1) =根，n(w，L(w)) = w .另外，对于任意内节点n，设ch(n)是n的任意固定子节点，若x为真，则设[ [x] ]为1，否则为-1。然后，分层softmax将p(wO|wI)定义如下:
    
![](https://ai-studio-static-online.cdn.bcebos.com/e2687ee339a749c2ac3b704cc05eb2c99f9d6de82f43463aa22a390e89882a65)

经过简单变化得到对数似然函数，然后使用随机梯度上升法可以进行最大化优化，从而学习参数，得到模型。

(详细推导可见http://www.hankcs.com/nlp/word2vec.html#respond)

根据单词的频率将单词组合在一起对于基于神经网络的语言模型来说是一种非常简单的加速技术


### 总结

基于Hierarchical Softmax的CBOW模型算法流程，梯度迭代使用了随机梯度上升法：

　　　　输入：基于CBOW的语料训练样本，词向量的维度大小M，CBOW的上下文大小2c,步长η
    
　　　　输出：霍夫曼树的内部节点模型参数θ，所有的词向量w
    
　　　　1. 基于语料训练样本建立霍夫曼树。

　　　　2. 随机初始化所有的模型参数θ，所有的词向量w
    
　　　　3. 进行梯度上升迭代过程

## 参考

[1]https://zhuanlan.zhihu.com/p/56139075

[2]http://www.hankcs.com/nlp/word2vec.html#respond

[3]https://www.cnblogs.com/pinard/p/7243513.html

[4]https://aistudio.baidu.com/aistudio/projectdetail/2198230

[5]Distributed Representations of Words and Phrases and their Compositionality()https://arxiv.org/abs/1310.4546v1
