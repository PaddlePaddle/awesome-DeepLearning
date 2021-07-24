# ShareBottom 多任务学习

## 概念：

共享底部结构的模型——多个任务共享隐藏层，每个任务接不通的输出层，即一个双塔模型（或者多塔模型），在共享底部结构中，学习相似性，每个独立的塔在学习一下每个label的自己独立的特性，这样设计很有意思。ShareBottom模型一个很重要的前提就是任务相关性，如果多个label的相关性比较差，那共享的底部结构就是一种相互负面影响，会对每个label的预测带来一定的负面影响。但是相似性问题本身就没有一个明确的度量，在全世界范围内都是一个开放问题，那么多相似性度量指标怎么选，为何这样能衡量相似性，这就陷入了另一个问题之中。


## 模型：

将id特征embedding，和dense特征concat一起，作为share bottom网络输入id

<img src="https://img-blog.csdnimg.cn/20200311123941650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2hvcml6b25oZWFydA==,size_16,color_FFFFFF,t_70" alt="å¨è¿éæå¥å¾çæè¿°" style="zoom: 50%;" />

特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。这在全站item变化比较快的情况下，这种方式就不是首选的方式。


## 作用：

shared-bottom是一种基于神经网络的多任务学习最常见的套路，底层公用若干层神经网络，然后上面每个任务自己特有的tower。

<img src="C:\Users\spade-卿\AppData\Roaming\Typora\typora-user-images\image-20210724180453257.png" alt="image-20210724180453257" style="zoom: 80%;" />

shared-bottm模型是神经网络多任务模型中出现较早，应用广泛的一种模型。如上图，不同的任务共享输入层、shared-bottom层，每个任务有自己的task-specific tower层，每个Tower层可以是一个独立的神经网络。在一些CV领域的多任务学习中，share-bottom层可能就是某种卷积模型。

## 场景：

多任务优化，即一个模型的产出可以同时对多个目标或任务进行推理，多目标之所以现在走红，是因为它一定程度上简化了我们对业务的细化分。例如，在电商场景中，传统的CTR模型，只会推理出用户要不要点，而不会对后续用户的是否下单有任何的关联性推理（如需推理则需要再做一个下单的模型），而多任务则不同，它不仅仅要推理出用户是否点击，还要推理出用户点击后是否会对商品下单，从而增加用户对商品的购买意愿，这样我们可以在同一个模型中推理出用户在点击后的并行行为，而不需要对点击场景和购买场景分开做处理，从而简化了业务场景分类。 



## 优缺点：

缺点：

模型的表达能力弱，泛化能力差

优点：

模型简单，使用范围广