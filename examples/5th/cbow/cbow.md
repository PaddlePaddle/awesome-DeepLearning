基于CBOW训练模型的word2vec

一、word2vec简介

word2vec是一种用于训练词向量的模型工具，作用是将所有词语投影到K维的向量空间，每个词语都可以用一个K维向量表示。
为什么要将词用向量来表示呢？这样可以给词语一个数学上的表示，使之可以适用于某些算法或数学模型。

二、CBOW介绍

Word2vec根据上下文之间的出现关系去训练词向量，有两种训练模式，Skip Gram和CBOW（constinuous bags of words），其中Skip Gram根据目标单词预测上下文，CBOW根据上下文预测目标单词，最后使用模型的部分参数作为词向量。本文中只介绍基于Hierarchical Softmax的CBOW训练模型，CBOW结构图如下。

![](https://ai-studio-static-online.cdn.bcebos.com/2f93de4d57ce41e6b414e5151b18868aaad13fb1971343c594a58b0bc8a2aab6)

第一层为输入层：包含context(w)中上下文的２×win（窗口）个词向量。即对应目标单词w，选取其上下文各win个单词的词向量作为输入。

第二层为投影层：将输入层的２×win个向量做累加求和。

第三层为输出层：对应一颗二叉树，叶子节点共Ｎ个，对应词典里的每个词。 我们是通过哈弗曼树来求得某个词的条件概率的。假设某个词ｗ，从根节点出发到ｗ这个叶子节点，中间会经过４词分支，每一次分支都可以视为一次二分类。从二分类来说，word2ecv定义分到左边为负类（编码为１），分到右边为正类（编码label为０）。在逻辑回归中，一个节点被分为正类的概率为ｐ，分为负类的概率为１－ｐ。将每次分类的结果累乘则得到p(w∣Context(w))。

概率p在逻辑回归二分类问题中，对于任意样本x=(x 
1
​	
 ,x 
2
​	
 ,x 
3
​	
 ,...,x 
n
​	
 ) 
T
 ，
利用sigmoid函数，求得分为正类的概率为h θ ( w ) = σ ( θ T x ) ，负类概率为1 − h θ ( w ) = σ ( θ T x )
 
sigmoidhan函数如图：

![](https://ai-studio-static-online.cdn.bcebos.com/d4f590b94674430fb0868fca6cdec2a5cafcfe10273d413f9043c00e3e9be5cb)

小结：对于词典D中的任意词w，Huffman树中必存在一条从根节点到词w对应结点的路径p w
 （且路径唯一），p w
 路径上存在个l w − 1分支，每个分支看做一个二分类，每一次分类产生一个概率，将这些概率乘起来，就是所需的p ( w ∣ C o n t e x t ( w ) )。
 
 ![](https://ai-studio-static-online.cdn.bcebos.com/e9c02adff3eb4d33a56193d70a9ce6476c635c271d1d4c968ec3c9c92f476acf)
 
d j w为1或0，即某个单词在某条分支上的huffman编码
 
x w为输入词向量的求和平均

θ j − 1 w为非叶子节点向量

由上可知我们的目标函数就是p ( w ∣ C o n t e x t ( w ) )，但在基于神经网络的语言模型的目标函数通常取为如下对数似然函数：

ζ= 
wϵc
∑
​	
 logp（w∣Context(w)）

训练过程就是要将目标函数最大化，word2vec采用随机梯度上升的方法。

三、训练过程

Step1:准备好语料，将训练数据保存为txt文件中。

另取一些数据作为测试数据。

Step2:设置一个类class，保存词以及它的哈夫曼树路径、哈弗曼编码、词频。

Step3:初始化各类参数，扫描语料库，统计词频，并依据每个词的词频生成生成哈弗曼树。生成哈弗曼树后生成每个词的哈弗曼编码以及路径。初始化输入层词向量syn0以及哈弗曼树上非叶子结点的向量syn1。

Step4:训练，迭代优化。 训练过程中就是通过不断的输入，用随机梯度上升的方法，去更新词向量的值（syn0），非叶子结点处向量的值(syn1)。实质上就是让词向量在词向量空间中找到正确的位置。
训练伪代码如图：

![](https://ai-studio-static-online.cdn.bcebos.com/4f550c6c3fe3489194ecfd2e4aa7c1ee2ad314a06cd647d39f02b59f89d77b21)

详细解释：

1、初始化neule=0（映射层反向求导参数）

2、将第一层词向量v(.)，syn0相加求平均得x w

3、

3.1首先获得关键字路径编码label(即d j w)

3.2将x w 与syn1（即θ j − 1 w）做乘积的结果再经sigmoid函数得概率q

3.3学习率η ，d j w为label（即编码0/1）。利用二分类的公式求得g

3.4更新e

3.5然后更新syn1。( 1 − d j w − q ) × θ j − 1 w即为求得的梯度，通过乘与学习率η再加上 θ j − 1 w来更新syn1。

4、然后更新syn0

Step5：训练完成后进行预测

将目标词上下文本作为输入，获得词向量后相加平均得x w， 求关键字路径哈夫曼编码，
将x w与syn1（ θ j − 1 w）做乘积的结果再经sigmoid函数得概率q

根据二分类，求得概率P。

累乘即可得目标词概率，如下公式

![](https://ai-studio-static-online.cdn.bcebos.com/4ddc3d34284a4c2cb4ff87cc9869c7224c52d45b149d4ab69e28728ff60a7d08)



