### **FastText模型**

一般情况下，使用fastText进行文本分类的同时也会产生词的embedding，即embedding是fastText分类的产物

（1）字符级别的n-gram

word2vec把语料库中的每个单词当成原子，它会为每个单词生成一个向量，这忽略了单词内部的形态特征，如“apple”与“apples”，两个单词都有较多的公共字符，即它们的内部形态类似，但是在传统的word2vec中，这种单词内部形态信息因为它们被转换成不同的id丢失了

为了克服这个问题，fastText使用了字符级别的n-grams来表示一个单词，对于“apple”，假设n的取值为3，则它的trigram有：

"<ap","app","ppl","ple","le>"

其中<表示前缀，>表示后缀，我们可以使用这5个trigram的向量叠加来表示“apple”的词向量

优点：

对于低频词生成的词向量效果会更好，因为它们的n-gram可以和其他词共享；对于训练词库之外的单词，仍然可以构建它们的词向量，可以叠加它们的字符级别n-gram向量

（2）模型架构

fastText模型架构和word2vec的CBOW模型架构非常相似，下面就是fastText模型的架构图：


![img](https://img-blog.csdnimg.cn/20190123173954785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lhbmdmZW5nbGluZzEwMjM=,size_16,color_FFFFFF,t_70)

从上图可以看出来，fastText模型包括输入层、隐含层、输出层共三层。其中输入的是词向量，输出的是label，隐含层是对多个词向量的叠加平均。

1）CBOW的输入是目标单词的上下文，fastText的输入是多个单词及其n-gram特征，这些单词用来表示单个文档

2）CBOW的输入单词被one-hot编码过，fastText的输入特征时被embedding过

3）CBOW的输出是目标词汇，fastText的输出是文档对应的类标

（3）核心思想

将整篇文档的词及n-gram向量叠加平均得到文档向量，然后使用文档向量做softmax多分类

小技巧：字符级n-gram特征的引入以及分层softmax分类

