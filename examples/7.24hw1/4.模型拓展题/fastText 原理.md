### fastText 原理

fastText 方法包含三部分：模型架构、层次 Softmax 和 N-gram 特征 

fastText 模型输入一个词的序列（一段文本或者一句话)，输出这个词序列属于不同类别的概率。 

序列中的词和词组组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签。

 fastText 在预测标签时使用了非线性激活函数，但在中间层不使用非线性激活函数。 

 fastText 模型架构和 Word2Vec 中的 CBOW 模型很类似。不同之处在于，fastText 预测标签， 而 CBOW 模型预测中间词

![image-20210721165651867](C:\Users\MSI-PC\AppData\Roaming\Typora\typora-user-images\image-20210721165651867.png)

### 改善运算效率——softmax层级

对于有大量类别的数据集，fastText使用了一个分层分类器（而非扁平式架构）。不同的类别被整合进树形
结构中（想象下二叉树而非 list）。在某些文本分类任务中类别很多，计算线性分类器的复杂度高。为了改善运行时间，fastText 模型使用了层次 Softmax 技巧。层次 Softmax 技巧建立在哈弗曼编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。

![image-20210721165816582](C:\Users\MSI-PC\AppData\Roaming\Typora\typora-user-images\image-20210721165816582.png)

考虑到线性以及多种类别的对数模型，这大大减少了训练复杂性和测试文本分类器的时间。fastText 也利用了类别（class）不均衡这个事实（一些类别出现次数比其他的更多），通过使用 Huffman 算法建立用于表征类别的树形结构。因此，频繁出现类别的树形结构的深度要比不频繁出现类别的树形结构的深度要小，这也使得进一步的计算效率更高。

![image-20210721165853894](C:\Users\MSI-PC\AppData\Roaming\Typora\typora-user-images\image-20210721165853894.png)

## FastText的词向量表征

### FastText的N-gram特征

常用的特征是词袋模型。但词袋模型不能考虑词之间的顺序，因此 fastText 还加入了 N-gram 特征。
“我 爱 她” 这句话中的词袋模型特征是 “我”，“爱”, “她”。这些特征和句子 “她 爱 我” 的特征是一样的。
如果加入 2-Ngram，第一句话的特征还有 “我-爱” 和 “爱-她”，这两句话 “我 爱 她” 和 “她 爱 我” 就能区别开来了。当然，为了提高效率，我们需要过滤掉低频的 N-gram。
在 fastText 中一个低维度向量与每个单词都相关。隐藏表征在不同类别所有分类器中进行共享，使得文本信息在不同类别中能够共同使用。这类表征被称为词袋（bag of words）（此处忽视词序）。在 fastText中也使用向量表征单词 n-gram来将局部词序考虑在内，这对很多文本分类问题来说十分重要。
举例来说：fastText能够学会“男孩”、“女孩”、“男人”、“女人”指代的是特定的性别，并且能够将这些数值存在相关文档中。然后，当某个程序在提出一个用户请求（假设是“我女友现在在儿？”），它能够马上在fastText生成的文档中进行查找并且理解用户想要问的是有关女性的问题。



### FastText词向量优势

（1）适合大型数据+高效的训练速度：能够训练模型“在使用标准多核CPU的情况下10分钟内处理超过10亿个词汇”，特别是与深度模型对比，fastText能将训练时间由数天缩短到几秒钟。使用一个标准多核 CPU，得到了在10分钟内训练完超过10亿词汇量模型的结果。此外， fastText还能在五分钟内将50万个句子分成超过30万个类别。
（2）支持多语言表达：利用其语言形态结构，fastText能够被设计用来支持包括英语、德语、西班牙语、法语以及捷克语等多种语言。它还使用了一种简单高效的纳入子字信息的方式，在用于像捷克语这样词态丰富的语言时，这种方式表现得非常好，这也证明了精心设计的字符 n-gram 特征是丰富词汇表征的重要来源。FastText的性能要比时下流行的word2vec工具明显好上不少，也比其他目前最先进的词态词汇表征要好。

![image-20210721165949762](C:\Users\MSI-PC\AppData\Roaming\Typora\typora-user-images\image-20210721165949762.png)

（3）fastText专注于文本分类，在许多标准问题上实现当下最好的表现（例如文本倾向性分析或标签预测）。FastText与基于深度学习方法的Char-CNN以及VDCNN对比：

![image-20210721170026371](C:\Users\MSI-PC\AppData\Roaming\Typora\typora-user-images\image-20210721170026371.png)

（4）比word2vec更考虑了相似性，比如 fastText 的词嵌入学习能够考虑 english-born 和 british-born 之间有相同的后缀，但 word2vec 却不能。

### FastText词向量与word2vec对比


相似处：

图模型结构很像，都是采用embedding向量的形式，得到word的隐向量表达。
都采用很多相似的优化方法，比如使用Hierarchical softmax优化训练和预测中的打分速度。
不同处：

模型的输出层：word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是 分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用；
模型的输入层：word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；
两者本质的不同，体现在 h-softmax的使用：

Wordvec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax
也会生成一系列的向量，但最终都被抛弃，不会使用。
fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）