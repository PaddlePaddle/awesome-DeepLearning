# CBOW

CBOW模型根据某个中心词前后A个连续的词，来计算该中心词出现的概率，即用上下文预测目标词。模型结构简易示意图如下：

![img](https://img-blog.csdnimg.cn/20200108164120879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDc3MTUyMQ==,size_16,color_FFFFFF,t_70)

模型有三层，输入层，隐藏层（又叫投影层），输出层。上图模型的window=2，即在中心词前后各选两个连续的词作为其上下文。输入层的w(t-2),w(t-1),w(t+1),w(t+2)是中心词w(t)的上下文。

**接下来根据下图，走一遍CBOW的流程，推导一下各层矩阵维度的变化。**

![img](https://img-blog.csdnimg.cn/20200108164247335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDc3MTUyMQ==,size_16,color_FFFFFF,t_70)

原始语料词库（corpus）中有V个单词。滑动窗口window选为A，那总共选取的上下文词个数C=2A.
1.在输入层，输入的是多个上下文单词的one-hot。
(维度：因为corpus中有V个单词，所以每个单词的one-hot的维度1*V，那么在输入层输入的就是C个1*V的向量，所以在输入层，数据维度为C*V)
2.设定最终获得的词向量的维度为N，初始化输入层与隐藏层之间的权重矩阵w，w维度为V*N。上下文单词的one-hot(C*V)与网络的输入权重矩阵w(V*N)相乘，得到C个1*N的向量。把它们求和再平均，得到隐藏层向量h，维度为1*N.
3.初始化隐藏层与输出层之间的权重矩阵 w ,维度为N*V。
4.隐藏层向量h(1*N)与 w ′ {w\prime} w′(N*V)相乘，得到1*V的向量u， u = h ⋅ w ′ u = h \cdot w\prime u=h⋅w′。为了方便概率表示,将向量u经过softmax，此时向量softmax(u)的每一维代表语料中的一个单词。向量softmax(u)概率最大的位置所代表的单词为模型预测出的中间词。
5.上一步输出的1*V向量与groud truth中的one hot比较。训练的目的是最大化实际中心词出现的概率，基于此定义损失函数，通过最小化损失函数，采用梯度下降算法更新W和W’。当网络收敛则训练完成，此时矩阵W就是我们想要的词向量。如果我们想要知道语料中某个词的向量表示，就用这个词的one-hot乘以权重矩阵w，得到1*N的向量，这个向量就是我们要找的这个词的向量表示。

因为词的one-hot表示中只有一个位置是1，其余都是0，那么与w相乘后，得到的是w中的某一列向量。由于每个词语的 one-hot里面 1 的位置是不同的，所以不同词的one-hot与w相乘，得到的词向量是w中的不同的列向量。所以可见，权重矩阵中的每一列向量都能对应的、唯一的表示corpus中的每一个词。所以要的词向量就是神经网络输入层与隐藏层之间的权重矩阵