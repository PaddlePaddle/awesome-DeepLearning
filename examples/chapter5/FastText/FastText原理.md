# 思想

  - **将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类。** 这中间涉及到两个技巧：字符级n-gram特征的引入以及分层Softmax分类。叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。
  - 模型的前半部分，即从输入层输入到隐含层输出部分：生成用来表征文档的向量。**叠加构成这篇文档的所有词及n-gram的词向量，然后取平均。** 叠加词向量背后的思想就是传统的词袋法，即将文档看成一个由词构成的集合。
  - 模型的后半部分，即从隐含层输出到输出层输出部分：是一个softmax线性多类别分类器，分类器的输入是一个用来表征当前文档的向量。
  - 子词嵌入（subword embedding），使用字符级别的n-grams表示一个单词。
    - 例子：对于单词“book”，假设n的取值为3，则它的trigram有:**“<bo”, “boo”, “ook”, “ok>”** 其中，<表示前缀，>表示后缀。于是，我们可以用这些trigram来表示“book”这个单词。

# FastText原理
fastText方法包含三部分，模型架构，层次SoftMax和N-gram子词特征。

## 1、模型架构
fastText的架构和word2vec中的CBOW的架构类似，因为它们的作者都是Facebook的科学家Tomas Mikolov，而且确实fastText也算是word2vec所衍生出来的。

**CBOW的架构:输入的是w(t)的上下文2d个词，经过隐藏层后，输出的是w(t)。**
<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="image\CBOW.PNG">
<br>
<div style="color:orange;
display: inline-block;
color: #999;
padding: 2px;"></div>
</center>

word2vec将上下文关系转化为多分类任务，进而训练逻辑回归模型，这里的类别数量是 |V| 词库大小。通常的文本数据中，词库少则数万，多则百万，在训练中直接训练多分类逻辑回归并不现实。

word2vec中提供了两种针对大规模多分类问题的优化手段， negative sampling 和 hierarchical softmax。在优化中，negative sampling 只更新少量负面类，从而减轻了计算量。hierarchical softmax 将词库表示成前缀树，从树根到叶子的路径可以表示为一系列二分类器，一次多分类计算的复杂度从|V|降低到了树的高度。

fastText模型架构:
其中x1,x2,...,xN−1,xN表示一个文本中的n-gram向量，每个特征是词向量的平均值。这和前文中提到的cbow相似，cbow用上下文去预测中心词，而此处用全部的n-gram去预测指定类别。
<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="image\fasttext.PNG">
<br>
<div style="color:orange;
display: inline-block;
color: #999;
padding: 2px;"></div>
</center>

## 2、层次SoftMax
对于有大量类别的数据集，fastText使用了一个分层分类器（而非扁平式架构）。不同的类别被整合进树形结构中（想象下二叉树而非 list）。在某些文本分类任务中类别很多，计算线性分类器的复杂度高。为了改善运行时间，fastText 模型使用了层次 Softmax 技巧。层次 Softmax 技巧建立在哈弗曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。

fastText 也利用了类别（class）不均衡这个事实（一些类别出现次数比其他的更多），通过使用 Huffman 算法建立用于表征类别的树形结构。因此，频繁出现类别的树形结构的深度要比不频繁出现类别的树形结构的深度要小，这也使得进一步的计算效率更高。

<center>
<img style="border-radius: 0.3125em;
box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
src="image\softmax.PNG">
<br>
<div style="color:orange;
display: inline-block;
color: #999;
padding: 2px;"></div>
</center>

## 3、N-gram子词特征
fastText 可以用于文本分类和句子分类。不管是文本分类还是句子分类，我们常用的特征是词袋模型。但词袋模型不能考虑词之间的顺序，因此 fastText 还加入了 N-gram 特征。在 fasttext 中，每个词被看做是 n-gram字母串包。为了区分前后缀情况，"<"， ">"符号被加到了词的前后端。除了词的子串外，词本身也被包含进了 n-gram字母串包。以 where 为例，n=3 的情况下，其子串分别为
<wh, whe, her, ere, re>，以及其本身 。


# fastText与CBOW不同点

  - CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些特征用来表示单个文档；
  - CBOW的输入单词被one-hot编码过，fastText的输入特征是被embedding过；
  - CBOW的输出是目标词汇，fastText的输出是文档对应的类标。

  - **值得注意的是，fastText在输入时，将单词的字符级别的n-gram向量作为额外的特征；在输出时，fastText采用了分层Softmax，大大降低了模型训练时间。**

# fastText与Word2Vec的异同

  - 相同点

    - 图模型结构很像，都是采用embedding向量的形式，得到word的隐向量表达。
    - 都采用很多相似的优化方法，比如使用Hierarchical softmax优化训练和预测中的打分速度。

  - 不同点

    - 层次softmax：CBOW的叶子节点是词和词频，fasttext叶子节点里是类标和类标的频数。

    - word2vec的目的是得到词向量，该词向量最终是在输入层得到的，输出层对应的h-softmax也会生成一系列的向量，但是最终都被抛弃，不会使用。

      fastText则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label
