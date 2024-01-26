
# LSTM应用场景

LSTM是RNN的一个优秀的变种模型，继承了大部分RNN模型的特性，同时解决了梯度反传过程由于逐步缩减而产生的Vanishing Gradient问题。具体到语言处理任务中，LSTM非常适合用于处理与时间序列高度相关的问题，例如机器翻译、对话生成、编码\解码等。

虽然在分类问题上，至今看来以CNN为代表的前馈网络依然有着性能的优势，但是LSTM在长远的更为复杂的任务上的潜力是CNN无法媲美的。它更真实地表征或模拟了人类行为、逻辑发展和神经组织的认知过程。尤其从2014年以来，LSTM已经成为RNN甚至深度学习框架中非常热点的研究模型，得到大量的关注和研究。


# TextCNN

 ![Image text](https://img-blog.csdnimg.cn/img_convert/227f2e9d40724ac1acb6ef0589ef13a8.png)


# DPCNN

 ![Image text](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2pwZy9BenVYZmVJTnhqWG9JYWIwWWVIRXNDMUN0NnNxWHBqNTRoaWJpY3FOaWNWTFJoRW5NMlAyYXdBc1NmeDZpY3NaSlFsUUROZjJNTkUwaWNZUnRPV1lpY2dwWk9tUS82NDA?x-oss-process=image/format,png)

# NNLM

设计如下图网络结构，即“神经网络语言模型，NNLM”，用来做语言模型。
它的训练网络结构和RNN、LSTM、CNN相比很简单。学习任务是输入某个句子“Bert won't even give me fifteen dollars for the bus ride home”中单词Wt=“fifteen”前面的t-1个单词“Bert won't even give me”，要求网络正确预测单词“fifteen”，即最大化P(Wt=“fifteen”|W1,W2,…W(t-1);θ)。前面的单词Wi是用one-hot编码作为原始输入，再乘矩阵Q获得向量C(Wi)，再将每个单词的C(Wi)拼接，接上隐层，然后接softmax得到预测概率得知后续接哪个单词。

而C(Wi)是什么？C(Wi)=Wi*Q，是单词对应的word embedding值，假设word embedding的维度为d、词典大小为V，矩阵Q的结构是V×d，每一行对应一个单词的embedding。Q也是一个待学习参数，用随机值初始化矩阵Q后开始训练、微调；所以NNLM任务不仅能够根据上文预测后接单词是什么，同时得到一个副产品——矩阵Q，即word embedding。这就是 word embedding是如何学会？其实就是一个预测任务的中间的可学习参数。

 ![Image text](https://img-blog.csdnimg.cn/20200810144330114.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTE5NDky,size_16,color_FFFFFF,t_70)
