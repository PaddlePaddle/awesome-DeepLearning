# CNN-DSSM

## 概念

DSSM（Deep Structured Semantic Models）的原理是通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 DNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。DSSM 采用词袋模型（BOW），因此丧失了语序信息和上下文信息。针对 DSSM 词袋模型丢失上下文信息的缺点，CNN-DSSM应运而生， CNN-DSSM 与 DSSM 的区别主要在于输入层和表示层。

## 模型

#### 输入层

###### 英文：

CNN-DSSM 在DSSM基础上还在输入层增加了word-trigram

<img src="https://blog-10039692.file.myqcloud.com/1501555685228_6957_1501555686382.png" alt="img" style="zoom:67%;" />

​                                                                                                       图1

- word-n-gram层：是对输入做了一个获取上下文信息的窗口，图中是word-trigram，取连续的3个单词。

- Letter-trigram：是把上层的三个单词通过3个字母的形式映射到3w维，然后把3个单词连接起来成9w维的空间。

- Convolutional layer：是通过Letter-trigram层乘上卷积矩阵获得，是普通的卷积操作。

- Max-pooling：是把卷积结果经过池化操作。

- Semantic layer：是语义层，是池化层经过全连接得到的。

  

如上图所示，word-trigram其实就是一个包含了上下文信息的滑动窗口。举个例子：把<`s`> online auto body ... <`s`>这句话提取出前三个词<`s`> online auto，之后再分别对这三个词进行letter-trigram映射到一个 3 万维的向量空间里，然后把三个向量 concat 起来，最终映射到一个 9 万维的向量空间里。

###### 中文：

英文的处理方式（word-trigram letter-trigram）在中文中并不可取，因为英文中虽然用了 word-ngram 把样本空间拉成了百万级，但是经过 letter-trigram 又把向量空间降到可控级别，只有 3`*`30K（9 万）。而中文如果用 word-trigram，那向量空间就是百万级的了，显然还是字向量（1.5 万维）比较可控。

#### 表示层

CNN-DSSM 的表示层由一个卷积神经网络组成，如下图所示：

<img src="https://blog-10039692.file.myqcloud.com/1501555818817_3444_1501555820078.png" alt="img" style="zoom: 67%;" />

​                                                                                                           图2

#### 匹配层

Query 和 Doc 的语义相似性可以用这两个语义向量(128 维) 的 cosine 距离来表示：

![img](https://blog-10039692.file.myqcloud.com/1501555545519_4107_1501555546427.png)

通过softmax 函数可以把Query 与正样本 Doc 的语义相似性转化为一个后验概率：

![img](https://blog-10039692.file.myqcloud.com/1501555590842_9539_1501555591755.png)

其中 r 为 softmax 的平滑因子，D 为 Query 下的正样本，D-为 Query 下的负样本（采取随机负采样），D 为 Query 下的整个样本空  间。

在训练阶段，通过极大似然估计，我们最小化损失函数：

![img](https://blog-10039692.file.myqcloud.com/1501555602634_219_1501555603542.png)

残差会在表示层的 DNN 中反向传播，最终通过随机梯度下降（SGD）使模型收敛，得到各网络层的参数{Wi,bi}。

## 作用

该模型既可以用来**预测两个句子的语义相似度**，又可以**获得某句子的低维语义向量表达**。

## 场景

以搜索引擎和搜索广告为例，最重要的也最难解决的问题是语义相似度，这里主要体现在两个方面：召回和排序。

在**召回**时，传统的文本相似性如 BM25，无法有效发现语义类 Query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性。

在**排序**时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"。

**DSSM（Deep Structured Semantic Models）** 为计算语义相似度提供了一种思路。

## 优缺点

- 优点：CNN-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留。
- 缺点：CNN-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息，对于间隔较远的上下文信息，难以有效保留。
