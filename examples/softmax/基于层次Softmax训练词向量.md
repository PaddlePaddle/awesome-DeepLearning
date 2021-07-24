## 基于层次Softmax训练词向量

### 词向量

向量（Word embedding），又叫Word嵌入式自然语言处理（NLP）中的一组语言建模和特征学习技术的统称，其中来自词汇表的单词或短语被映射到实数的向量。 从概念上讲，它涉及从每个单词一维的空间到具有更低维度的连续向量空间的数学嵌入。

生成这种映射的方法包括神经网络，单词共生矩阵的降维，概率模型，可解释的知识库方法，和术语的显式表示 单词出现的背景。

当用作底层输入表示时，单词和短语嵌入已经被证明可以提高NLP任务的性能，例如语法分析和情感分析。

### 层次Softmax

如下图的架构，在进行最优化的求解过程中：从隐藏层到输出的softmax层的计算量很大，因为要计算所有词的softmax概率，再去找概率最大的值。

1） cbow的对周围词的调整是统一的：求出的gradient的值会同样的作用到每个周围词的词向量当中去。那么，cbow预测行为的次数跟整个文本的词数几乎是相等的（每次预测行为才会进行一次backpropgation, 而往往这也是最耗时的部分），复杂度大概是O(V);

2）Skip-gram 中，每个词在作为中心词时，都要使用周围词进行预测一次。这样相当于比cbow的方法多进行了K次（假设K为窗口大小），因此时间的复杂度为O(KV)，训练时间要比cbow要长。

![Softmax](images\Softmax.webp)

### **基于Hierarchical Softmax的模型**: 

**输入层到隐层**：所有输入词向量求和并取平均的方法。

**隐层到输出层**：从隐藏层到输出的softmax层这里的计算量个改进，也就是层次softmax。由于我们把之前所有都要计算的从输出softmax层的概率计算变成了一颗二叉霍夫曼树，那么我们的softmax概率计算只需要沿着树形结构进行就可以了。如下图所示，我们可以沿着霍夫曼树从根节点一直走到我们的叶子节点的词w<sub>2</sub>。

  和之前的神经网络语言模型相比，我们的霍夫曼树的所有**内部节点**就类似之前神经网络隐藏层的**神经元**,其中，**根节点的词向量对应我们的投影后的词向量**，而所有**叶子节点**就类似于之前神经网络softmax**输出层的神经元**，叶子节点的个数就是词汇表的大小。在霍夫曼树中，隐藏层到输出层的softmax映射不是一下子完成的，而是沿着霍夫曼树一步步完成的，因此这种softmax取名为"Hierarchical Softmax"。

  如何“沿着霍夫曼树一步步完成”呢？在word2vec中，我们采用了**二元逻辑回归**的方法，即规定沿着左子树走，那么就是负类(霍夫曼树编码1)，沿着右子树走，那么就是正类(霍夫曼树编码0)。判别正类和负类的方法是使用sigmoid函数。使用霍夫曼树的好处：首先，由于是二叉树，之前计算量为V,现在变成了**log2V**。第二，由于使用霍夫曼树是高频的词靠近树根，这样高频词需要更少的时间会被找到，这符合我们的贪心优化思想。

  因此，目标：我们的目标就是找到合适的**所有节点的词向量**和**所有内部节点θ**, 使训练样本达到**最大似然**。

### 基于Hierarchical Softmax的模型梯度计算

我们使用最大似然法来寻找所有节点的词向量和所有内部节点θ。先拿上面的w<sub>2</sub>例子来看，我们期望最大化下面的似然函数：
$$
\prod_{i=1}^{3} P\left(n\left(w_{i}\right), i\right)=\left(1-\frac{1}{1+e^{-x_{w}^{T} \theta_{1}}}\right)\left(1-\frac{1}{1+e^{-x_{u}^{T} \theta_{2}}}\right) \frac{1}{1+e^{-x_{U}^{T} \theta_{3}}}
$$
对于所有的训练样本，我们期望最大化所有样本的似然函数乘积。

　　　　为了便于我们后面一般化的描述，我们定义输入的词为w,其从输入层词向量求和平均后的霍夫曼树根节点词向量为x<sub>w</sub>, 从根节点到w所在的叶子节点，包含的节点总数为l<sub>w</sub>, w在霍夫曼树中从根节点开始，经过的第ii个节点表示为p<sub>i</sub><sup>w</sup>,对应的霍夫曼编码为d<sub>i</sub><sup>w</sup>∈{0,1}d<sub>i</sub><sup>w</sup>∈{0,1},其中i=2,3,...l<sub>i</sub><sup>w</sup>=2,3,...l<sub>w</sub>。而该节点对应的模型参数表示为θ<sub>i</sub><sup>w</sup>, 其中i=1,2,...l<sub>w</sub>−1，没有i=l<sub>w</sub>是因为模型参数仅仅针对于霍夫曼树的内部节点。

　　　　定义w经过的霍夫曼树某一个节点j的逻辑回归概率为P(d<sub>j</sub><sup>w</sup>|x<sub>w</sub>,θ<sub>j-1</sub><sup>w</sup>)P(d<sub>j</sub><sup>w</sup>|x<sub>w</sub>,θ<sub>j-1</sub><sup>w</sup>)，其表达式为：
$$
P\left(d_{j}^{w} \mid x_{w}, \theta_{j-1}^{w}\right)=\left\{\begin{array}{ll}
\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right) & d_{j}^{w}=0 \\
1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right) & d_{j}^{w}=1
\end{array}\right.
$$
那么与某一个目标输出词w，其最大似然为：
$$
\prod_{j=2}^{l_{w}} P\left(d_{j}^{w} \mid x_{w}, \theta_{j-1}^{w}\right)=\prod_{j=2}^{l_{w}}\left[\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]^{1-d_{j}^{w}}\left[1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]^{d_{j}^{w}}
$$
在word2vec中，由于使用的是随机梯度上升法，所以并没有把所有样本的似然乘起来得到真正的训练集最大似然，仅仅每次只用一个样本更新梯度，这样做的目的是减少梯度计算量。这样我们可以得到w的对数似然函数L如下：
$$
L=\log \prod_{j=2}^{l_{w}} P\left(d_{j}^{w} \mid x_{w}, \theta_{j-1}^{w}\right)=\sum_{j=2}^{l_{w}}\left(\left(1-d_{j}^{w}\right) \log \left[\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]+d_{j}^{w} \log \left[1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right]\right)
$$
要得到模型中w词向量和内部节点的模型参数θ, 我们使用梯度上升法即可。首先我们求模型参数θ<sub>j-1</sub><sup>w</sup>的梯度：
$$
\begin{aligned}
\frac{\partial L}{\partial \theta_{j-1}^{w}} &=\left(1-d_{j}^{w}\right) \frac{\left(\sigma ( x _ { w } ^ { T } \theta _ { j - 1 } ^ { w } ) \left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right.\right.}{\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} x_{w}-d_{j}^{w} \frac{\left(\sigma ( x _ { w } ^ { T } \theta _ { j - 1 } ^ { w } ) \left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right.\right.}{1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)} x_{w} \\
&=\left(1-d_{j}^{w}\right)\left(1-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) x_{w}-d_{j}^{w} \sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right) x_{w} \\
&=\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) x_{w}
\end{aligned}
$$
同样的方法，可以求出x<sub>w</sub>的梯度表达式如下：
$$
\frac{\partial L}{\partial x_{w}}=\sum_{j=2}^{l_{w}}\left(1-d_{j}^{w}-\sigma\left(x_{w}^{T} \theta_{j-1}^{w}\right)\right) \theta_{j-1}^{w}
$$
有了梯度表达式，我们就可以用梯度上升法进行迭代来一步步的求解我们需要的所有的参数。