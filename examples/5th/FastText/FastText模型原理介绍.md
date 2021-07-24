# FastText模型原理介绍


---
##一、fastText简介
fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：
1、fastText在保持高精度的情况下加快了训练速度和测试速度
2、fastText不需要预训练好的词向量，fastText会自己训练词向量
3、fastText两个重要的优化：Hierarchical Softmax、N-gram


##二、fastText模型架构
fastText模型架构:其中x1,x2,…,xN−1,xN表示一个文本中的n-gram向量，每个特征是词向量的平均值。和word2vec中的CBOW的架构类似，因为它们的作者都是Facebook的科学家Tomas Mikolov，而且确实fastText也算是words2vec所衍生出来的。

Continuous Bog-Of-Words：
![此处输入图片的描述][1]
fastText
![此处输入图片的描述][2]

从上图可以看出来，fastText模型包括输入层、隐含层、输出层共三层。其中输入的是词向量，输出的是label，隐含层是对多个词向量的叠加平均。

1）CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些单词用来表示单个文档

2）CBOW的输入单词被one-hot编码过，fastText的输入特征时被embedding过

3）CBOW的输出是目标词汇，fastText的输出是文档对应的类标

fastText 模型输入一个词的序列（一段文本或者一句话)，输出这个词序列属于不同类别的概率。

序列中的词和词组组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签。

fastText 在预测标签时使用了非线性激活函数，但在中间层不使用非线性激活函数。fastText 模型架构和 Word2Vec 中的 CBOW 模型很类似。不同之处在于，fastText 预测标签，而 CBOW 模型预测中间词。

###2.2 层次SoftMax

对于有大量类别的数据集，fastText使用了一个分层分类器（而非扁平式架构）。不同的类别被整合进树形结构中（想象下二叉树而非 list）。在某些文本分类任务中类别很多，计算线性分类器的复杂度高。为了改善运行时间，fastText 模型使用了层次 Softmax 技巧。层次 Softmax 技巧建立在哈弗曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。

fastText 也利用了类别（class）不均衡这个事实（一些类别出现次数比其他的更多），通过使用 Huffman 算法建立用于表征类别的树形结构。因此，频繁出现类别的树形结构的深度要比不频繁出现类别的树形结构的深度要小，这也使得进一步的计算效率更高。

###![此处输入图片的描述][3]

2.3 N-gram特征

fastText 可以用于文本分类和句子分类。不管是文本分类还是句子分类，我们常用的特征是词袋模型。但词袋模型不能考虑词之间的顺序，因此 fastText 还加入了 N-gram 特征。“我 爱 她” 这句话中的词袋模型特征是 “我”，“爱”, “她”。这些特征和句子 “她 爱 我” 的特征是一样的。如果加入 2-Ngram，第一句话的特征还有 “我-爱” 和 “爱-她”，这两句话 “我 爱 她” 和 “她 爱 我” 就能区别开来了。当然啦，为了提高效率，我们需要过滤掉低频的 N-gram。

##3.核心思想

将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类

##4.模型泛用分析
总的来说，fastText的学习速度比较快，效果还不错。fastText适用与分类类别非常大而且数据集足够多的情况，当分类类别比较小或者数据集比较少的话，很容易过拟合。

可以完成无监督的词向量的学习，可以学习出来词向量，来保持住词和词之间，相关词之间是一个距离比较近的情况；
也可以用于有监督学习的文本分类任务，（新闻文本分类，垃圾邮件分类、情感分析中文本情感分析，电商中用户评论的褒贬分析）




  [1]: https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=538653864,60800681&fm=173&app=25&f=JPEG?w=600&h=723&s=0CAE74334116DFCE4CF555CE000010B0
  [2]: https://img-blog.csdn.net/20180206120020822?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvam9obl9iaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast
  [3]: https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=1678164792,2485250115&fm=173&app=25&f=JPEG?w=640&h=405&s=C910E01A110244E404C9A4D20000C0B1
  [4]: https://user-images.githubusercontent.com/86996619/125740192-17532ce4-a1b7-4cff-ac56-c37bcbcbb0db.png
  [5]: https://www.zhihu.com/equation?tex=%20%5Chat%7B%5Cmathbf%7Ba%7D%7D%5El%20=%20%5Cfrac%7B%5Cmathbf%7Ba%7D%5El%20-%20%5Cmu%5El%7D%7B%5Csqrt%7B%28%5Csigma%5El%29%5E2%20%2b%20%5Cepsilon%7D%7D%20%5Ctag2
  [6]: https://www.zhihu.com/equation?tex=%5Cepsilon
  [7]: https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D%5El%20=%20f%28%5Cmathbf%7Bg%7D%5El%20%5Codot%20%5Chat%7B%5Cmathbf%7Ba%7D%7D%5El%20%2b%20%5Cmathbf%7Bb%7D%5El%29%20%5Ctag3
  [8]: https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D%20=%20f%28%5Cfrac%7B%5Cmathbf%7Bg%7D%7D%7B%5Csqrt%7B%5Csigma%5E2%20%2b%20%5Cepsilon%7D%7D%20%5Codot%20%28%5Cmathbf%7Ba%7D%20-%20%5Cmu%29%2b%20%5Cmathbf%7Bb%7D%29%20%5Ctag4
  [9]: https://user-images.githubusercontent.com/86996619/125741499-7f211fc3-e4d4-4e4b-8f48-5e1f8209007c.png
  [10]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS00Y2YzYWFmNDdjYjhhNDFkLnBuZw?x-oss-process=image/format,png
  [11]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS1lYjQ2MWRjZDk2MWE3MzllLnBuZw?x-oss-process=image/format,png
  [12]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS0yZWM5OGUyZWE1OTQzYTQ3LnBuZw?x-oss-process=image/format,png
  [13]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS1mYWU4ZmZkM2M2MjUzMDVmLnBuZw?x-oss-process=image/format,png
  [14]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS0xZWFlYTgzYjdmNjJjMDcyLnBuZw?x-oss-process=image/format,png
  [15]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS1kN2FmYTZkYzA2NjU4NzkzLnBuZw?x-oss-process=image/format,png
  [16]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS1kZTNkOTkzZDAxNzI3ZmUxLnBuZw?x-oss-process=image/format,png
  [17]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS0zOTNiMGMyYzExMDg3YTViLnBuZw?x-oss-process=image/format,png
  [18]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS01NTFiNWNiYTZjNjQwYjU4LnBuZw?x-oss-process=image/format,png
  [19]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS02NGVkNmM3NDkwZGI2YTc3LnBuZw?x-oss-process=image/format,png
  [20]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS0yYTZjZDMyOTZiMzZlYTVkLnBuZw?x-oss-process=image/format,png
  [21]: https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xNTcxMzExNS02NTQxZmZkNzliMjc4ZTBhLnBuZw?x-oss-process=image/format,png