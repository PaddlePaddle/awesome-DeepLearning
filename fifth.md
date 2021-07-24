**##基于层次softmax训练词向量**
4.5.1 预处理：构建haffman树
根据语料中的每个word的词频构建赫夫曼tree，词频越高，则离树根越近，路径越短。如上图所示，词典V VV中的每个word都在叶子节点上，每个word需要计算两个信息：路径（经过的每个中间节点）以及赫夫曼编码，例如：
“respond”的路径经过的节点是（6，4，3），编码label是（1，0，0）
构建完赫夫曼tree后，每个叶子节点都有唯一的路径和编码，hierarchical softmax与softmax不同的是，在hierarchical softmax中，不对V VV中的word词进行向量学习，而是对中间节点进行向量学习，而每个叶子上的节点可以通过路径中经过的中间节点去表示。

4.5.2 模型的输入
输入部分，在cbow或者skip-gram模型，要么是上下文word对应的id词向量平均，要么是中心词对应的id向量，作为hidden层的输出向量

4.5.3 样本label
不同softmax的是，每个词word对应的是一个V VV大小的one-hot label，hierarchical softmax中每个叶子节点word，对应的label是赫夫曼编码，一般长度不超过，在训练的时候，每个叶子节点的label统一编码到一个固定的长度，不足的可以进行pad

4.5.4 训练过程
我们用一个例子来描述，假如一个训练样本如下：
![](https://ai-studio-static-online.cdn.bcebos.com/bf3678d57ec54ba386c909ac55a0080e531a722199024039980bf1aa5f21da6a)

假如我们用skip-gram模型，则第一部分，根据"chupacabra" 词的one-hot编码乘以W WW权重矩阵，得到“chupachabra”的词向量表示，也就是hidden的输出，根据目标词“active”从赫夫曼tree中得到它的路径path，即经过的节点（6，4，3），而这些中间节点的向量是模型参数需要学习的，共有V − 1 V-1V−1个向量，通过对应的节点id，取出相应的向量，假设（这里设词向量维度为N），分别与hidden输出向量相乘，再经过sigmoid函数，得到一个3*1的score分值，与实际label [1,0,1], 通过如下公式：
![](https://ai-studio-static-online.cdn.bcebos.com/4ba484b3031d4d6587f6ca76ab683148a14573a121934c8d9099ca889994d465)

由于小于1的分值相乘，会容易溢出，则取log，乘法变加法，loss则是与我们期望的概率值越大相反，需要取负用tf.negative实现，计算出loss，则就可以用优化器来计算模型中各参数的梯度进行更新了

4.5.5 模型预测
训练过程中，每个word我们都提前知道了赫夫曼编码，所以训练的时候我们平均大概只要计算中间节点，就可以得到结果。但是在预测阶段，由于我们需要计算每个叶子节点输出概率值，然后取最大的一个概率，所以在预测阶段并不能省时间。那么怎么计算每个叶子节点也就是word的概率值？
从输入层到hidden层，得到输出向量，然后分别计算与每个叶子节点score分值。与训练过程一样，hidden向量与叶子节点的path经过的中间节点向量相乘，再sigmoid，然后用如下公式：
![](https://ai-studio-static-online.cdn.bcebos.com/9b927905e3144c2aa2b47e560e823738755d2ea6520044579f9c0108a4be9ecd)

计算所有叶子节点的score分值，取分值最大的则就是预测的word label


**##LSTM可以实现的其他类型NLP任务##**
词性标注，命名实体识别
![](https://ai-studio-static-online.cdn.bcebos.com/a9230c28fac541fe8830502b3351b0c78931176427634f6eb0c8fe1bb1ee02a7)


用于生成文本，比如语音识别、机器翻译等
![](https://ai-studio-static-online.cdn.bcebos.com/a637a1a376f74c29aa0cc50ba57f2d456ab83de695594e538c56a81aa365def7)


用于问答系统，机器翻译等
![](https://ai-studio-static-online.cdn.bcebos.com/6e4ca1ccc1b149528bc8f3771315ecdc66d1455ddc82444aa8a0ba8e9cf69481)


**##CBOW原理**

如 **图5** 所示，CBOW是一个具有3层结构的神经网络，分别是：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/72397490c0ba499692cff31484431c57bc9d20f7ef344454868e12d628ec5bd3" width="400" ></center>
<center><br>图5：CBOW的算法实现</br></center>
<br></br>

* **输入层：** 一个形状为C×V的one-hot张量，其中C代表上线文中词的个数，通常是一个偶数，我们假设为4；V表示词表大小，我们假设为5000，该张量的每一行都是一个上下文词的one-hot向量表示，比如“Pineapples, are, and, yellow”。
* **隐藏层：** 一个形状为V×N的参数张量W1，一般称为word-embedding，N表示每个词的词向量长度，我们假设为128。输入张量和word embedding W1进行矩阵乘法，就会得到一个形状为C×N的张量。综合考虑上下文中所有词的信息去推理中心词，因此将上下文中C个词相加得一个1×N的向量，是整个上下文的一个隐含表示。
* **输出层：** 创建另一个形状为N×V的参数张量，将隐藏层得到的1×N的向量乘以该N×V的参数张量，得到了一个形状为1×V的向量。最终，1×V的向量代表了使用上下文去推理中心词，每个候选词的打分，再经过softmax函数的归一化，即得到了对中心词的推理概率：

$$𝑠𝑜𝑓𝑡𝑚𝑎𝑥({O_i})= \frac{exp({O_i})}{\sum_jexp({O_j})}$$

如 **图6** 所示，Skip-gram是一个具有3层结构的神经网络，分别是：

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/a572953b845d4c91bdf6b7b475e7b4437bee69bd60024eb2b8c46f56adf2bdef" width="400" ></center>
<center><br>图6：Skip-gram算法实现</br></center>
<br></br>

- **Input Layer（输入层）**：接收一个one-hot张量 $V \in R^{1 \times \text{vocab\_size}}$ 作为网络的输入，里面存储着当前句子中心词的one-hot表示。
- **Hidden Layer（隐藏层）**：将张量$V$乘以一个word embedding张量$W_1 \in R^{\text{vocab\_size} \times \text{embed\_size}}$，并把结果作为隐藏层的输出，得到一个形状为$R^{1 \times \text{embed\_size}}$的张量，里面存储着当前句子中心词的词向量。
- **Output Layer（输出层）**：将隐藏层的结果乘以另一个word embedding张量$W_2 \in R^{\text{embed\_size} \times \text{vocab\_size}}$，得到一个形状为$R^{1 \times \text{vocab\_size}}$的张量。这个张量经过softmax变换后，就得到了使用当前中心词对上下文的预测结果。根据这个softmax的结果，我们就可以去训练词向量模型。

在实际操作中，使用一个滑动窗口（一般情况下，长度是奇数），从左到右开始扫描当前句子。每个扫描出来的片段被当成一个小句子，每个小句子中间的词被认为是中心词，其余的词被认为是这个中心词的上下文。



**##LSTM预测##**
1.预处理：词转换为词向量
2.创建模型和验证：将输入映射到输出的收敛-发散模型（convergent-divergent）
3.预测：最优词预测

**##FastText##**
一、fastText简介
fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：
1、fastText在保持高精度的情况下加快了训练速度和测试速度
2、fastText不需要预训练好的词向量，fastText会自己训练词向量
3、fastText两个重要的优化：Hierarchical Softmax、N-gram

二、fastText模型架构
fastText模型架构和word2vec中的CBOW很相似， 不同之处是fastText预测标签而CBOW预测的是中间词，即模型架构类似但是模型的任务不同。下面我们先看一下CBOW的架构：
![](https://ai-studio-static-online.cdn.bcebos.com/126b1e219ba647b79ccb8f46cb1a067889445f30c4f84fae9d0492c4e49e0ff1)
word2vec将上下文关系转化为多分类任务，进而训练逻辑回归模型，这里的类别数量|V|词库大小。通常的文本数据中，词库少则数万，多则百万，在训练中直接训练多分类逻辑回归并不现实。word2vec中提供了两种针对大规模多分类问题的优化手段， negative sampling 和hierarchical softmax。在优化中，negative sampling 只更新少量负面类，从而减轻了计算量。hierarchical softmax 将词库表示成前缀树，从树根到叶子的路径可以表示为一系列二分类器，一次多分类计算的复杂度从|V|降低到了树的高度

fastText模型架构:其中x1,x2,…,xN−1,xN表示一个文本中的n-gram向量，每个特征是词向量的平均值。这和前文中提到的cbow相似，cbow用上下文去预测中心词，而此处用全部的n-gram去预测指定类别
![](https://ai-studio-static-online.cdn.bcebos.com/c9bae2caaffc4bb2baa81d8fb2da817feec73f982564405190529af72a9f17ed)

