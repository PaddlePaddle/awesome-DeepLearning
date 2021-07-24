四、ShareBottom多任务学习

1、概念：共享参数得过拟合几率比较低，hard parameter网络每个任务有自己的参数，模型参数之间的距离会作为正则项来保证参数尽可能相似。hard模式对于关联性比较强的任务，降低网络过拟合，提升了泛化效果。常用的是share bottom 的网络 可以共享参数，提高泛化

2、模型：

将id特征embedding，和dense特征concat一起，作为share bottom网络输入，id

![image-20210724224349316](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224349316.png)

特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。这在全站item变化比较快的情况下，这种方式就不是首选的方式。3、作用：多任务

4、场景：商场推荐系统

5、优缺点