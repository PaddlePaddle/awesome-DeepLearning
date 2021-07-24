**深度学习基础知识：**
**1.CNN-DSSM概念：**
DSSM（Deep Structured Semantic Models）的原理很简单，通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义向量表达。
**2.CNN-DSSM模型：**
DSSM 从下往上可以分为三层结构：输入层、表示层、匹配层：
![](https://ai-studio-static-online.cdn.bcebos.com/46db12d09ad74c618fc377ad6b98311d4ac176b4a0fa4127ae7dd46da289b3bb)

针对 DSSM 词袋模型丢失上下文信息的缺点，CLSM（convolutional latent semantic model）应运而生，又叫 CNN-DSSM。CNN-DSSM 与 DSSM 的区别主要在于输入层和表示层。
**2.1 输入层**
（1）英文
英文的处理方式，除了letter-trigram，CNN-DSSM 还在输入层增加了word-trigram
![](https://ai-studio-static-online.cdn.bcebos.com/2f2af54a5e8e45abae387621e9cd212b661858e775bf4e97b8ae353752265a3c)
如上图所示，word-trigram其实就是一个包含了上下文信息的滑动窗口。举个例子：把<s> online auto body ... <s>这句话提取出前三个词<s> online auto，之后再分别对这三个词进行letter-trigram映射到一个 3 万维的向量空间里，然后把三个向量 concat 起来，最终映射到一个 9 万维的向量空间里。
（2）中文
英文的处理方式（word-trigram letter-trigram）在中文中并不可取，因为英文中虽然用了 word-ngram 把样本空间拉成了百万级，但是经过 letter-trigram 又把向量空间降到可控级别，只有 3*30K（9 万）。而中文如果用 word-trigram，那向量空间就是百万级的了，显然还是字向量（1.5 万维）比较可控。
**2.2 表示层**
CNN-DSSM 的表示层由一个卷积神经网络组成，如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/4b70517c7eb74c9cb557d6b8babd898f0bbd246d4f6f4046bd7fddd41faadb03)
 （1）卷积层——Convolutional layer
卷积层的作用是提取滑动窗口下的上下文特征。以下图为例，假设输入层是一个 302*90000（302 行，9 万列）的矩阵，代表 302 个字向量（query 的和 Doc 的长度一般小于 300，这里少了就补全，多了就截断），每个字向量有 9 万维。而卷积核是一个 3*90000 的权值矩阵，卷积核以步长为 1 向下移动，得到的 feature map 是一个 300*1 的矩阵，feature map 的计算公式是(输入层维数 302-卷积核大小 3 步长 1)/步长 1=300。而这样的卷积核有 300 个，所以形成了 300 个 300*1 的 feature map 矩阵。
![](https://ai-studio-static-online.cdn.bcebos.com/9e27c555893e4dbfa17bfbcbca57522d90d296ed70114ec28f734de582517d9a)
  （2）池化层——Max pooling layer
池化层的作用是为句子找到全局的上下文特征。池化层以 Max-over-time pooling 的方式，每个 feature map 都取最大值，得到一个 300 维的向量。Max-over-pooling 可以解决可变长度的句子输入问题（因为不管 Feature Map 中有多少个值，只需要提取其中的最大值）。不过我们在上一步已经做了句子的定长处理（固定句子长度为 302），所以就没有可变长度句子的问题。最终池化层的输出为各个 Feature Map 的最大值，即一个 300*1 的向量。这里多提一句，之所以 Max pooling 层要保持固定的输出维度，是因为下一层全链接层要求有固定的输入层数，才能进行训练。
（3）全连接层——Semantic layer
最后通过全连接层把一个 300 维的向量转化为一个 128 维的低维语义向量。全连接层采用 tanh 函数：
![](https://ai-studio-static-online.cdn.bcebos.com/35935b6adf934ffaa06136b90fe60c6596b9732e5e1b496da3c99c1c511baed4)
  
**2.3 匹配层**
 Query 和 Doc 的语义相似性可以用这两个语义向量(128 维) 的 cosine 距离来表示：
![](https://ai-studio-static-online.cdn.bcebos.com/5d122db4fd9243b49e322f2ed6ea400f6dfcea78444f43e485a63ebccc25f535)
通过softmax 函数可以把Query 与正样本 Doc 的语义相似性转化为一个后验概率：
![](https://ai-studio-static-online.cdn.bcebos.com/dee47c7073264ed38135593605f6108377ce504daaa941cabaee356c310112e5)
其中 r 为 softmax 的平滑因子，D 为 Query 下的正样本，D-为 Query 下的负样本（采取随机负采样），D 为 Query 下的整个样本空间。

在训练阶段，通过极大似然估计，我们最小化损失函数：
![](https://ai-studio-static-online.cdn.bcebos.com/9b73576642014b69805c82578d488455fb75536e45734de49363fb64c2707975)
残差会在表示层的 DNN 中反向传播，最终通过随机梯度下降（SGD）使模型收敛，得到各网络层的参数{Wi,bi}。

**3.CNN-DSSM作用：**
  DSSM（Deep Structured Semantic Models）的原理很简单，通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。该模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义向量表达。
  DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。针对 DSSM 词袋模型丢失上下文信息的缺点，CLSM（convolutional latent semantic model）应运而生，又叫 CNN-DSSM。CNN-DSSM 与 DSSM 的区别主要在于输入层和表示层。

**4.CNN-DSSM场景**
以搜索引擎和搜索广告为例，最重要的也最难解决的问题是语义相似度，这里主要体现在两个方面：召回和排序。
在召回时，传统的文本相似性如 BM25，无法有效发现语义类 query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性。
在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"。
DSSM（Deep Structured Semantic Models）为计算语义相似度提供了一种思路。

**5.CNN-DSSM优缺点**
优点：CNN-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留。

缺点：对于间隔较远的上下文信息，难以有效保留。举个例子，I grew up in France... I speak fluent French，显然 France 和 French 是具有上下文依赖关系的，但是由于 CNN-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息。
