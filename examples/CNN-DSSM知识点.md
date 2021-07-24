# 一、CNN-DSSM知识点补充
针对 DSSM 词袋模型丢失上下文信息的缺点，CLSM[2]（convolutional latent semantic model）应运而生，又叫 CNN-DSSM。CNN-DSSM 与 DSSM 的区别主要在于输入层和表示层。

**3.1 输入层**

（1）英文

英文的处理方式，除了上文提到的 letter-trigram，CNN-DSSM 还在输入层增加了word-trigram
![](https://ai-studio-static-online.cdn.bcebos.com/0f19953c6f9b47349b64f87b5bbd934b12ed67e1cd884dc6a9f4ae0ee95134d7)
如上图所示，word-trigram其实就是一个包含了上下文信息的滑动窗口。举个例子：把<s> online auto body ... <s>这句话提取出前三个词<s> online auto，之后再分别对这三个词进行letter-trigram映射到一个 3 万维的向量空间里，然后把三个向量 concat 起来，最终映射到一个 9 万维的向量空间里。

（2）中文

英文的处理方式（word-trigram letter-trigram）在中文中并不可取，因为英文中虽然用了 word-ngram 把样本空间拉成了百万级，但是经过 letter-trigram 又把向量空间降到可控级别，只有 3*30K（9 万）。而中文如果用 word-trigram，那向量空间就是百万级的了，显然还是字向量（1.5 万维）比较可控。

**3.2 表示层**
CNN-DSSM 的表示层由一个卷积神经网络组成，如下图所示：
  ![](https://ai-studio-static-online.cdn.bcebos.com/741af869e95d4ecfa297e9134ba969d99e42475ce0364af5823a91ce1c79f58b)
  （1）卷积层——Convolutional layer

卷积层的作用是提取滑动窗口下的上下文特征。以下图为例，假设输入层是一个 302*90000（302 行，9 万列）的矩阵，代表 302 个字向量（query 的和 Doc 的长度一般小于 300，这里少了就补全，多了就截断），每个字向量有 9 万维。而卷积核是一个 3*90000 的权值矩阵，卷积核以步长为 1 向下移动，得到的 feature map 是一个 300*1 的矩阵，feature map 的计算公式是(输入层维数 302-卷积核大小 3 步长 1)/步长 1=300。而这样的卷积核有 300 个，所以形成了 300 个 300*1 的 feature map 矩阵。
  ![](https://ai-studio-static-online.cdn.bcebos.com/7bdcf96e200d4db9b1c94f3ffb0e99d366ae8b843a0a43bdbcb264e82c08435e)
  （2）池化层——Max pooling layer

池化层的作用是为句子找到全局的上下文特征。池化层以 Max-over-time pooling 的方式，每个 feature map 都取最大值，得到一个 300 维的向量。Max-over-pooling 可以解决可变长度的句子输入问题（因为不管 Feature Map 中有多少个值，只需要提取其中的最大值）。不过我们在上一步已经做了句子的定长处理（固定句子长度为 302），所以就没有可变长度句子的问题。最终池化层的输出为各个 Feature Map 的最大值，即一个 300*1 的向量。这里多提一句，之所以 Max pooling 层要保持固定的输出维度，是因为下一层全链接层要求有固定的输入层数，才能进行训练。

（3）全连接层——Semantic layer

最后通过全连接层把一个 300 维的向量转化为一个 128 维的低维语义向量。全连接层采用 tanh 函数：
  ![](https://ai-studio-static-online.cdn.bcebos.com/602441fedbbd42639e282ce9a70d1b6729bf816c9301466089944295029cce6a)
  3.3 匹配层
CNN-DSSM 的匹配层和 DSSM 的一样，这里省略。

3.4 优缺点
优点：CNN-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留。

缺点：对于间隔较远的上下文信息，难以有效保留。举个例子，I grew up in France... I speak fluent French，显然 France 和 French 是具有上下文依赖关系的，但是由于 CNN-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息。
