**CBOW用于神经网络语言模型**

 在word2vec出现之前，已经有用神经网络DNN来用训练词向量进而处理词与词之间的关系了。采用的方法一般是一个三层的神经网络结构（当然也可以多层），分为输入层，隐藏层和输出层(softmax层)。

 这个模型是如何定义数据的输入和输出呢？一般分为CBOW(Continuous Bag-of-Words 与Skip-Gram两种模型。

 CBOW模型的训练输入是某一个特征词的上下文相关的词对应的词向量，而输出就是这特定的一个词的词向量。比如下面这段话，我们的上下文大小取值为4，特定的这个词是"Learning"，也就是我们需要的输出词向量,上下文对应的词有8个，前后各4个，这8个词是我们模型的输入。由于CBOW使用的是词袋模型，因此这8个词都是平等的，也就是不考虑他们和我们关注的词之间的距离大小，只要在我们上下文之内即可。

![](https://github.com/zbt57/images/blob/main/01.png)

 这样我们这个CBOW的例子里，我们的输入是8个词向量，输出是所有词的softmax概率（训练的目标是期望训练样本特定词对应的softmax概率最大），对应的CBOW神经网络模型输入层有8个神经元，输出层有词汇表大小个神经元。隐藏层的神经元个数我们可以自己指定。通过DNN的反向传播算法，我们可以求出DNN模型的参数，同时得到所有的词对应的词向量。这样当我们有新的需求，要求出某8个词对应的最可能的输出中心词时，我们可以通过一次DNN前向传播算法并通过softmax激活函数找到概率最大的词对应的神经元即可。

## CBOW的算法实现

对比Skip-gram，CBOW和Skip-gram的算法实现如下图

![img](https://ai-studio-static-online.cdn.bcebos.com/eee9dc52cd4f4be5b74c568df2e302859be16460fca44960aa2d788ea8b9328c)


图2：CBOW和Skip-gram的算法实现

- **Input Layer（输入层）**：接收one-hot张量 V∈R^1×vocab_sizeV  作为网络的输入，里面存储着当前句子中上下文单词的one-hot表示。
- **Hidden Layer（隐藏层）**：将张量V乘以一个word embedding张量W1∈Rvocab_size×embed_size并把结果作为隐藏层的输出，得到一个形状为R1×embed_size的张量，里面存储着当前句子上下文的词向量。
- **Output Layer（输出层）**：将隐藏层的结果乘以另一个word embedding张量W2∈Rembed_size×vocab_size得到一个形状为R1×vocab_size的张量。这个张量经过softmax变换后，就得到了使用当前上下文对中心的预测结果。根据这个softmax的结果，我们就可以去训练词向量模型。

在实际操作中，使用一个滑动窗口（一般情况下，长度是奇数），从左到右开始扫描当前句子。每个扫描出来的片段被当成一个小句子，每个小句子中间的词被认为是中心词，其余的词被认为是这个中心词的上下文。

CBOW算法和skip-gram算法最本质的区别就是：**CBOW算法是以上下文预测中心词，而skip-gram算法是以中心城预测上下文。**

