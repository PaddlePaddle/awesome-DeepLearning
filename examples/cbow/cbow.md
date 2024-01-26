# CBOW

* word2vec简介
* CBOW介绍
* 训练过程

## word2vec简介

​	word2vec是一种用于训练词向量的模型工具，作用是将所有词语投影到K维的向量空间，每个词语都可以用一个K维向量表示。
​	为什么要将词用向量来表示呢？这样可以给词语一个数学上的表示，使之可以适用于某些算法或数学模型。

## CBOW介绍

Word2vec根据上下文之间的出现关系去训练词向量，有两种训练模式，Skip Gram和CBOW（constinuous bags of words），其中Skip Gram根据目标单词预测上下文，CBOW根据上下文预测目标单词，最后使用模型的部分参数作为词向量。本文中只介绍基于Hierarchical Softmax的CBOW训练模型，CBOW结构图如下。


![cbow](C:\Users\SongWood\shujiaxuexi\baidushixi\NeuralNetDemo\Word2Vec\images\cbow.png)

第一层为输入层：包含context(w)中上下文的２×win（窗口）个词向量。即对应目标单词w，选取其上下文各win个单词的词向量作为输入。
第二层为投影层：将输入层的２×win个向量做累加求和。
第三层为输出层：对应一颗二叉树，叶子节点共Ｎ个，对应词典里的每个词。 我们是通过哈弗曼树来求得某个词的条件概率的。假设某个词ｗ，从根节点出发到ｗ这个叶子节点，中间会经过４词分支，每一次分支都可以视为一次二分类。从二分类来说，word2ecv定义分到左边为负类（编码为１），分到右边为正类（编码label为０）。在逻辑回归中，一个节点被分为正类的概率为ｐ，分为负类的概率为１－ｐ。将每次分类的结果累乘则得到$p(w \mid \operatorname{Context}(w))$。
sigmoid函数如下图所示：

![sigmoid](C:\Users\SongWood\shujiaxuexi\baidushixi\NeuralNetDemo\Word2Vec\images\sigmoid.png)

​	对于词典D中的任意词w，Huffman树中必存在一条从根节点到词w对应结点的路径$p^w$（且路径唯一），$p^w$路径上存在个$l^w-1$分支，每个分支看做一个二分类，每一次分类产生一个概率，将这些概率乘起来，就是所需的$p(w \mid \operatorname{Context}(w))$。

