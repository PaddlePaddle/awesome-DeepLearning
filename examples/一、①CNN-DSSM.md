CNN-DSSM

概念

CNN-DSSM在DSSM的基础上改进了数据的预处理和深度，通过搜索引擎里 Query 和 Title 的海量的点击曝光日志，用 CNN 把 Query 和 Title 表达为低纬语义向量，并通过 cosine 距离来计算两个语义向量的距离，最终训练出语义相似度模型。

模型

CNN-DSSM架构图如图所示

![CNN_DSSM1](C:\Users\apple\Desktop\image\CNN_DSSM1.png)

输入：\(Query\)是代表用户输入，\(document\)是数据库中的文档。

- word-n-gram层：是对输入做了一个获取上下文信息的窗口，图中是word-trigram，取连续的3个单词。
- Letter-trigram：是把上层的三个单词通过3个字母的形式映射到3w维，然后把3个单词连接起来成9w维的空间。
- Convolutional layer：是通过Letter-trigram层乘上卷积矩阵获得，是普通的卷积操作。
- Max-pooling：是把卷积结果经过池化操作。
- Semantic layer：是语义层，是池化层经过全连接得到的。

可以发现CNN-DSSM和DNN-DSSM基本流程是差不多的，就是用卷积和池化的操作代替了DNN的操作。

作用

CNN-DSSM表示层由一个卷积神经网络组成，如图所示。

![1501555818817_3444_1501555820078](C:\Users\apple\Desktop\image\1501555818817_3444_1501555820078.png)

卷积层（Convolutional layer）的作用是提取滑动窗口下的上下文特征。

池化层（Max pooling layer）的作用是为句子找到全局的上下文特征。池化层以 Max-over-time pooling 的方式，每个 feature map 都取最大值，得到一个 300 维的向量。Max-over-pooling 可以解决可变长度的句子输入问题。

全连接层（Semantic layer）采用tanh 函数，把一个 300 维的向量转化为一个 128 维的低维语义向量。

![1501555912876_4680_1501555913803](C:\Users\apple\Desktop\image\1501555912876_4680_1501555913803.png)

场景

CNN-DSSM模型既可以用来预测两个句子的语义相似度，又可以获得某句子的低纬语义向量表达。

优缺点

- 优点：CNN-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留，克服了 DSSM 词袋模型丢失上下文信息的缺点。
- 缺点：CNN-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息，对于间隔较远的上下文信息，难以有效保留。