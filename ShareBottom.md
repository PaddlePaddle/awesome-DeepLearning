四、ShareBottom
    将id特征embedding，和dense特征concat一起，作为share bottom网络输入，id特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界
落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的
embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。
这在全站item变化比较快的情况下，这种方式就不是首选的方式。
