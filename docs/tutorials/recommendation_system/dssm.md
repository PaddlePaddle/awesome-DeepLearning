# DSSM

## 背景

以搜索引擎和搜索广告为例，最重要的也最难解决的问题是语义相似度，这里主要体现在两个方面：召回和排序。
在召回时，传统的文本相似性如 BM25，无法有效发现语义类 query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性。
在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"。
DSSM（Deep Structured Semantic Models）为计算语义相似度提供了一种思路。

## DSSM
DSSM（Deep Structured Semantic Models）的原理很简单，通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义向量表达。

DSSM 从下往上可以分为三层结构：输入层、表示层、匹配层

![dssm](https://raw.githubusercontent.com/w5688414/paddleImage/main/dssm_img/dssm.png)

### 输入层
输入层做的事情是把句子映射到一个向量空间里并输入到 DNN 中，这里英文和中文的处理方式有很大的不同。

#### 英文

英文的输入层处理方式是通过word hashing。举个例子，假设用 letter-trigams 来切分单词（3 个字母为一组，#表示开始和结束符），boy 这个单词会被切为 #-b-o, b-o-y, o-y-#
	![word hashing](https://raw.githubusercontent.com/w5688414/paddleImage/main/dssm_img/word_hashing.png)
这样做的好处有两个：首先是压缩空间，50 万个词的 one-hot 向量空间可以通过 letter-trigram 压缩为一个 3 万维的向量空间。其次是增强范化能力，三个字母的表达往往能代表英文中的前缀和后缀，而前缀后缀往往具有通用的语义。

这里之所以用 3 个字母的切分粒度，是综合考虑了向量空间和单词冲突：



<table>
	<tr>
	    <th></th>
	    <th colspan="2">Letter-Bigram</th>
	    <th colspan="2">Letter-Trigram</th>  
	</tr >
	<tr >
	    <td>word Size</td>
	    <td>Token Size</td>
	     <td>Collision</td>
	      <td>Token Size</td>
	      <td> Collision </td>
	</tr>
	<tr>
	    <td>40k</td>
	    <td >1107</td>
	    <td >18</td>
	    <td >10306</td>
	    <td >2</td>
	</tr>
	<tr>
	    <td>500k</td>
	    <td >1607</td>
	    <td >1192</td>
	    <td >30621</td>
	    <td >22</td>
	</tr>
</table>

如上表，以 50 万个单词的词库为例，2 个字母的切分粒度的单词冲突为 1192（冲突的定义：至少有两个单词的 letter-bigram 向量完全相同），而 3 个字母的单词冲突降为 22 效果很好，且转化后的向量空间 3 万维不是很大，综合考虑选择 3 个字母的切分粒度。

#### 中文

中文的输入层处理方式与英文有很大不同，首先中文分词是个让所有 NLP 从业者头疼的事情，即便业界号称能做到 95%左右的分词准确性，但分词结果极为不可控，往往会在分词阶段引入误差。所以这里我们不分词，而是仿照英文的处理方式，对应到中文的最小粒度就是单字了。

由于常用的单字为 1.5 万左右，而常用的双字大约到百万级别了，所以这里出于向量空间的考虑，采用字向量（one-hot）作为输入，向量空间约为 1.5 万维。

### 表示层
DSSM 的表示层采用 BOW（Bag of words）的方式，相当于把字向量的位置信息抛弃了，整个句子里的词都放在一个袋子里了，不分先后顺序。
紧接着是一个含有多个隐层的 DNN，如下图所示：

![representation](https://raw.githubusercontent.com/w5688414/paddleImage/main/dssm_img/representaion_layer.png)

用$W_{i}$ 表示第 i 层的权值矩阵，$b_{i}$表示第 i 层的偏置项。则第一隐层向量 l2（300 维），第 二个隐层向量 l3（300 维），输出向量 y（128 维）,用数学公式可以分别表示为：

$$l_{1}=W_{1}x$$
$$l_{i}=f(W_{i}l_{i-1}+b_{i}) ,i=2,...,N-1$$
$$y=f(W_{N}l_{N-1}+b_{N})$$

用 tanh 作为隐层和输出层的激活函数：

$$f(x)=\frac{1-e^{-2x}}{1+e^{-2x}}$$
最终输出一个 128 维的低纬语义向量。

### 匹配层
Query 和 Doc 的语义相似性可以用这两个语义向量(128 维) 的 cosine 距离来表示：

$$R(Q,D)=cosine(y_{Q},y_{D})=\frac{y_{Q}^Ty_{D}}{||y_{Q}|| ||y_{D}||}$$

通过softmax 函数可以把Query 与正样本 Doc 的语义相似性转化为一个后验概率：

$$P(D^{+}|Q)=\frac{exp(\gamma R(Q,D^{+}))}{\sum_{D^{'}\in D}exp(\gamma R(Q,D^{'}))}$$

其中 r 为 softmax 的平滑因子，D 为 Query 下的正样本，D-为 Query 下的负样本（采取随机负采样），D 为 Query 下的整个样本空间。

在训练阶段，通过极大似然估计，我们最小化损失函数：

$$L(\Lambda)=-log \prod_{(Q,D^{+})}P(D^{+}|Q)$$

残差会在表示层的 DNN 中反向传播，最终通过随机梯度下降（SGD）使模型收敛，得到各网络层的参数$\{W_{i},b_{i}\}$。

负样本出现在计算softmax中，loss反向传播只用正样本。

### 优缺点
+ 优点：DSSM 用字向量作为输入既可以减少切词的依赖，又可以提高模型的泛化能力，因为每个汉字所能表达的语义是可以复用的。另一方面，传统的输入层是用 Embedding 的方式（如 Word2Vec 的词向量）或者主题模型的方式（如 LDA 的主题向量）来直接做词的映射，再把各个词的向量累加或者拼接起来，由于 Word2Vec 和 LDA 都是无监督的训练，这样会给整个模型引入误差，DSSM 采用统一的有监督训练，不需要在中间过程做无监督模型的映射，因此精准度会比较高。

+ 缺点：上文提到 DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。另一方面，DSSM 采用弱监督、端到端的模型，预测结果不可控。

## 参考文献

[1]. Huang P S, He X, Gao J, et al. Learning deep structured semantic models for web search using clickthrough data[C]// ACM International Conference on Conference on Information & Knowledge Management. ACM, 2013:2333-2338.




