# 基于层次softmax的优化策略

2013年，Mikolov提出的经典word2vec算法就是通过上下文来学习语义信息。word2vec包含两个经典模型：CBOW（Continuous Bag-of-Words）和Skip-gram。CBOW通过上下文的词向量推理中心词。而Skip-gram则根据中心词推理上下文。

输出时需要结果softmax函数归一化，得到对中心词的推理概率
$$
𝑠𝑜𝑓𝑡𝑚𝑎𝑥({O_i})= \frac{exp({O_i})}{\sum_jexp({O_j})}
$$
从上面的公式可以看出，softmax分母那项归一化，每次需要计算所有的O的输出值，才可以得到当前j节点的输出，当V很大的时候，O(V)的计算代价会非常高。所以在训练word2vec模型的时候，用到了两个tricks，一个是negative sampling，每次只采少量的负样本，不需要计算全部的V；

另外一个是hierarchical softmax，通过构建赫夫曼tree来做层级softmax，从复杂度O(V)降低到O(log_2V)

### **hierachical softmax**

#### 1、哈夫曼树

哈夫曼树( Huffman tree) 又称最优二叉树，是种带权路径最短的树，其特点是**权重越大，叶子节点就越靠近根节点，即权重越大，叶子节点搜索路径越短**

根据这个特性构造出的层次Softmax能够缩短目标类别的搜索路径。

![img](https://img-blog.csdnimg.cn/20200517121425833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0JHb29kSGFiaXQ=,size_16,color_FFFFFF,t_70)



首先对所有在V词表的词，根据词频来构建哈夫曼tree，词频越大，路径越短，编码信息更少。tree中的所有的叶子节点构成了词V，中间节点则共有V-1个，上面的每个叶子节点存在唯一的从根到该节点的path，如上图所示，词**w_2**的path n(**w_2**,1 ) ,n(**w_2**,2) , n(**w_3**,3) 其中n(w,j)表示词w的path的第j个节点。

#### 2、叶子节点词的概率表示

上图假设我们需要计算**w_2**的输出概率，我们定义从根节点开始，每次经过中间节点，做一个二分类任务（左边或者右边），所以我们定义中间节点的n左边概率为：
$$
p(n,left) = σ({v_n^{'}}^T.h)
$$
其中v_n'是中间节点的向量，右边概率：
$$
p(n, right)=1-σ({v_n^{'}}^T.h)= σ(-{v_n^{'}}^T.h)
$$
从根节点到**w_2**，我们可以计算概率值为：
$$
p(w_2=w_O)=
p(n(w_2,1),left).p(n(w_2,2),left).p(n(w_3,3),right)=
σ({v_{n(w_2,1)}^{'}}^T.h). σ({v_{n(w_2,2)}^{'}}^T.h). σ(-{v_{n(w_3,3)}^{'}}^T.h)
$$
其中 σ为sigmoid函数

#### 3、 各叶子节点概率值相加为1

可以得出
$$
\sum_{i=1}^{V}p(w_i=w_O) = 1
$$

### 训练

 **1、预处理：构建haffman树**
根据语料中的每个word的词频构建赫夫曼tree，词频越高，则离树根越近，路径越短。如上图所示，词典V VV中的每个word都在叶子节点上，每个word需要计算两个信息：路径（经过的每个中间节点）以及赫夫曼编码，构建完赫夫曼tree后，每个叶子节点都有唯一的路径和编码，hierarchical softmax与softmax不同的是，在hierarchical softmax中，不对V中的word词进行向量学习，而是对中间节点进行向量学习，而每个叶子上的节点可以通过路径中经过的中间节点去表示。

**2 模型的输入**
输入部分，在cbow或者skip-gram模型，要么是上下文word对应的id词向量平均，要么是中心词对应的id向量，作为hidden层的输出向量

**3 样本label**
不同softmax的是，每个词word对应的是一个V大小的one-hot label，hierarchical softmax中每个叶子节点word，对应的label是赫夫曼编码，一般长度不超过 log_2V，在训练的时候，每个叶子节点的label统一编码到一个固定的长度，不足的可以进行pad



