# TextCNN

## 1、**Yoon Kim**在论文[(2014 EMNLP) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)提出TextCNN。

将**卷积神经网络CNN**应用到**文本分类**任务，利用**多个不同size的kernel**来提取句子中的关键信息（类似于多窗口大小的ngram**）**，从而能够更好地捕捉局部相关性。

![20190326101508877](C:\Users\asus\Desktop\20190326101508877.png)

![20190326102103235](C:\Users\asus\Desktop\20190326102103235.png)

上图为模型的框架。假设我们有一些句子需要对其进行分类。句子中每个词是由n维词向量组成的，也就是说输入矩阵大小为m*n，其中m为句子长度。

CNN需要对输入样本进行卷积操作，**对于文本数据，filter不再横向滑动，仅仅是向下移动**，有点类似于N-gram在提取词与词间的局部相关性。图中共有三种步长策略，分别是2,3,4，每个步长都有两个filter（实际训练时filter数量会很多）。在不同词窗上应用不同filter，最终得到6个卷积后的向量。然后对每一个向量进行最大化池化操作并拼接各个池化值，最终得到这个句子的特征表示，将这个句子向量丢给分类器进行分类，至此完成整个流程。

### **嵌入层（Embedding Layer）**

通过一个隐藏层, 将 one-hot 编码的词投影到一个低维空间中，本质上是特征提取器，在指定维度中编码语义特征。 这样, 语义相近的词, 它们的欧氏距离或余弦距离也比较近。（作者使用的单词向量是预训练的，方法为fasttext得到的单词向量，当然也可以使用word2vec和GloVe方法训练得到的单词向量）。

### **卷积层（Convolution Laye）**

在处理图像数据时，CNN使用的卷积核的宽度和高度的一样的，但是在text-CNN中，卷积核的宽度是与词向量的维度一致。这是因为我们输入的每一行向量代表一个词，在抽取特征的过程中，词做为文本的最小粒度。而高度和CNN一样，可以自行设置（通常取值2,3,4,5），高度就类似于n-gram了。由于输入是一个句子，句子中相邻的词之间关联性很高，因此，当用卷积核进行卷积时，不仅考虑了词义而且考虑了词序及其上下文（类似于skip-gram和CBOW模型的思想）。

### **池化层（Pooling Layer）**

因为在卷积层过程中使用了不同高度的卷积核，使通过卷积层后得到的向量维度不一致，所以在池化层中，使用1-Max-pooling对每个特征向量池化成一个值，即抽取每个特征向量的最大值表示该特征，而且认为这个最大值表示最重要的特征。当对所有特征向量进行1-Max-Pooling后，将每个值给拼接起来。得到池化层最终的特征向量。在池化层到全连接层之前可以加上dropout防止过拟合。

### **全连接层（Fully connected layer）**

全连接层跟其他模型一样，假设有两层全连接层，第一层可以加上`ReLU`作为激活函数，第二层则使用softmax激活函数得到属于每个类的概率。

### **TextCNN的小变种**

在词向量构造方面可以有以下不同的方式：

CNN-rand: 随机初始化每个单词的词向量通过后续的训练去调整。

CNN-static: 使用预先训练好的词向量，如word2vec训练出来的词向量，在训练过程中不再调整该词向量。 

CNN-non-static: 使用预先训练好的词向量，并在训练过程进一步进行调整。

CNN-multichannel: 将static与non-static作为两通道的词向量。

### **参数与超参数**

- **sequence_length** ：对句子做定长处理, 比如定为n, 超过的截断, 不足的补0
- **num_classes** ：多分类, 分为几类
- **vocabulary_size** ：语料库的词典大小, 记为|D|
- **embedding_size** ：将词向量的维度, 由原始的 |D| 降维到 embedding_size
- **filter_size_arr** ：多个不同size的filter

## 2.2015年“A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification”论文详细地阐述了关于TextCNN模型的调参心得。



![1182656-20180919171920103-1233770993](C:\Users\asus\Desktop\1182656-20180919171920103-1233770993.png)

### TextCNN详细过程：

- **Embedding**：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
- **Convolution**：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
- **MaxPolling**：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
- **FullConnection and Softmax**：最后接一层全连接的 softmax 层，输出每个类别的概率。



### 通道：

- 图像中可以利用 (R, G, B) 作为不同channel；
- 文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法。

### 一维卷积（conv-1d）：

- 图像是二维数据；
- **文本是一维数据，因此在TextCNN卷积用的是一维卷积**（在**word-level**上是一维卷积；虽然文本经过词向量表达后是二维数据，但是在embedding-level上的二维卷积没有意义）。一维卷积带来的问题是需要**通过设计不同 kernel_size 的 filter 获取不同宽度的视野**

### Pooling层：

利用CNN解决文本分类问题的文章很多，比如这篇 [A Convolutional Neural Network for Modelling Sentences](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1404.2188.pdf) 将 pooling 改成 **(dynamic) k-max pooling** ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息。

比如在情感分析场景，举个例子：

```
“我觉得这个地方景色还不错，但是人也实在太多了”
```

 虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用 k-max pooling能够很好捕捉这类信息。

### 论文调参结论

- 使用预训练的word2vec 、 GloVe初始化效果会更好。一般不直接使用One-hot。
- 卷积核的大小影响较大，一般取1~10，对于句子较长的文本，则应选择大一些。
- 卷积核的数量也有较大的影响，一般取100~600 ，同时一般使用Dropout（0~0.5）。
- 激活函数一般选用ReLU 和 tanh。
- 池化使用1-max pooling。
- 随着feature map数量增加，性能减少时，试着尝试大于0.5的Dropout。
- 评估模型性能时，使用交叉验证。
  

 