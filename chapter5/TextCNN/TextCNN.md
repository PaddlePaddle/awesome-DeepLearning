## TextCNN：一个基于卷积的文本分类模型

### TextCNN

对于文本分类问题，常见的方法无非就是抽取文本的特征，比如使用doc2evc或者LDA模型将文本转换成一个固定维度的特征向量，然后在基于抽取的特征训练一个分类器。 然而研究证明，TextCnn在文本分类问题上有着更加卓越的表现。从直观上理解，TextCNN通过一维卷积来获取句子中n-gram的特征表示。TextCNN对文本浅层特征的抽取能力很强，在短文本领域如搜索、对话领域专注于意图分类时效果很好，应用广泛，且速度快，一般是首选；对长文本领域，TextCNN主要靠filter窗口抽取特征，在长距离建模方面能力受限，且对语序不敏感。

CNN可以识别出当前任务中具有预言性的n元语法（且如果使用特征哈希可以使用无约束的n元语法词汇，同时保持词嵌入矩阵的约束）；CNN卷积结构还允许有相似成分的n元语法分享预测行为，即使在预测过程中遇见未登录的特定n元语法；层次化的CNN每层有效着眼于句子中更长的n元语法，使得模型还可以对非连续n元语法敏感。

### TextCNN结构

《[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)》[1]模型示意图

![](\images\1.png)

《[A Sensitivity Analysis ](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1510.03820.pdf)...》[2]模型示意图（TextCNN）

![](\images\2.png)

### 嵌入层(embedding layer)

textcnn使用预先训练好的词向量作embedding layer。对于数据集里的所有词，因为每个词都可以表征成一个向量，因此我们可以得到一个嵌入矩阵MM, MM里的每一行都是词向量。这个MM可以是静态(static)的，也就是固定不变。可以是非静态(non-static)的，也就是可以根据反向传播更新。
CNN-rand

 作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。

CNN-static

 模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。

CNN-non-static

 同（2），只是训练的时候，Embedding layer跟随整个网络一起训练。

CNN-multichannel

 Embedding layer有两个channel，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。

### 卷积池化层(convolution and pooling)

#### 卷积(convolution)

输入一个句子，首先对这个句子进行切词，假设有s个单词。对每个词，跟句嵌入矩阵M, 可以得到词向量。假设词向量一共有d维。那么对于这个句子，便可以得到s行d列的矩阵AϵRs×d. 
        我们可以把矩阵A看成是一幅图像，使用卷积神经网络去提取特征。由于句子中相邻的单词关联性总是很高的，因此可以使用一维卷积，即文本卷积与图像卷积的不同之处在于只在文本序列的一个方向（垂直）做卷积，卷积核的宽度固定为词向量的维度d。高度是超参数，可以设置。 对句子单词每个可能的窗口做卷积操作得到特征图(feature map) c = [c_1, c_2, …, c_s-h+1]。

现在假设有一个卷积核，是一个宽度为d，高度为h的矩阵w，那么w有h∗d个参数需要被更新。对于一个句子，经过嵌入层之后可以得到矩阵AϵRs×d。 A[i:j]表示A的第i行到第j行, 那么卷积操作可以用如下公式表示：

 叠加上偏置b,在使用激活函数f激活, 得到所需的特征。公式如下：

对一个卷积核，可以得到特征cϵRs−h+1, 总共s−h+1个特征。我们可以使用更多高度h不同的卷积核，得到更丰富的特征表达。

Note: 

1 TextCNN网络包括很多不同窗口大小的卷积核，常用的filter size ∈ {3,4,5}，每个filter的feature maps=100。这里的特征图就是不同的k元语法。如上图中分别有两个不同的二、三和四元语法。

如果设置padding='same'即使用宽卷积，则每个feature maps for each region size都是seq_len*1，所有的feature map可以拼接成seq_len*(num_filters*num_filter_size)的矩阵，回到输入类似维度，这样就可以使用多层cnn了。

2 通道（Channels）：图像中可以利用 (R, G, B) 作为不同channel。而文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法；channel也可以一个是词序列，另一个channel是对应的词性序列。接下来就可以通过加和或者拼接进行结合。

#### 池化(pooling)

 不同尺寸的卷积核得到的特征(feature map)大小也是不一样的，因此我们对每个feature map使用池化函数，使它们的维度相同。

Max Pooling

 最常用的就是1-max pooling，提取出feature map照片那个的最大值，通过选择每个feature map的最大值，可捕获其最重要的特征。这样每一个卷积核得到特征就是一个值，对所有卷积核使用1-max pooling，再级联起来，可以得到最终的特征向量，这个特征向量再输入softmax layer做分类。这个地方可以使用drop out防止过拟合。

![](\images\3.jpg)

CNN中采用Max Pooling操作有几个好处：首先，这个操作可以保证特征的位置与旋转不变性，因为不论这个强特征在哪个位置出现，都会不考虑其出现位置而能把它提出来。但是对于NLP来说，这个特性其实并不一定是好事，因为在很多NLP的应用场合，特征的出现位置信息是很重要的，比如主语出现位置一般在句子头，宾语一般出现在句子尾等等。     其次，MaxPooling能减少模型参数数量，有利于减少模型过拟合问题。因为经过Pooling操作后，往往把2D或者1D的数组转换为单一数值，这样对于后续的Convolution层或者全联接隐层来说无疑单个Filter的参数或者隐层神经元个数就减少了。        再者，对于NLP任务来说，可以把变长的输入X整理成固定长度的输入。因为CNN最后往往会接全联接层，而其神经元个数是需要事先定好的，如果输入是不定长的那么很难设计网络结构。

 但是，CNN模型采取MaxPooling Over Time也有缺点：首先特征的位置信息在这一步骤完全丢失。在卷积层其实是保留了特征的位置信息的，但是通过取唯一的最大值，现在在Pooling层只知道这个最大值是多少，但是其出现位置信息并没有保留；另外一个明显的缺点是：有时候有些强特征会出现多次，出现次数越多说明这个特征越强，但是因为Max Pooling只保留一个最大值，就是说同一特征的强度信息丢失了。

Average Pooling

average pooling即取每个维度的均值而不是最大值。理解是对句子中的连续词袋(CBOW)而不是词进行卷积得到的表示（lz：每个filter都是对cbow来的）。

K-Max Pooling

取所有特征值中得分在Top –K的值，并（保序拼接）保留这些特征值原始的先后顺序（即多保留一些特征信息供后续阶段使用）。[A Convolutional Neural Network for Modelling Sentences]

![](C:\Users\VULCAN\Desktop\基础知识\TextCNN\images\4.jpg)

Note: pooling 改成 k-max pooling ，pooling阶段保留 k 个最大的信息，保留了全局的序列信息。比如在情感分析场景，举个例子：“ 我觉得这个地方景色还不错，但是人也实在太多了 ”。虽然前半部分体现情感是正向的，全局文本表达的是偏负面的情感，利用 k-max pooling能够很好捕捉这类信息。

Dynamic Pooling之Chunk-MaxPooling

把某个Filter对应的Convolution层的所有特征向量进行分段，切割成若干段后，在每个分段里面各自取得一个最大特征值，比如将某个Filter的特征向量切成3个Chunk，那么就在每个Chunk里面取一个最大值，于是获得3个特征值。因为是先划分Chunk再分别取Max值的，所以保留了比较粗粒度的模糊的位置信息；当然，如果多次出现强特征，则也可以捕获特征强度。至于这个Chunk怎么划分，可以有不同的做法，比如可以事先设定好段落个数，这是一种静态划分Chunk的思路；也可以根据输入的不同动态地划分Chunk间的边界位置，可以称之为动态Chunk-Max方法。Event Extraction via Dynamic Multi-Pooling Convolutional Neural Networks这篇论文提出的是一种ChunkPooling的变体，就是动态Chunk-Max Pooling的思路，实验证明性能有提升。Local Translation Prediction with Global Sentence Representation 这篇论文也用实验证明了静态Chunk-Max性能相对MaxPooling Over Time方法在机器翻译应用中对应用效果有提升。

![](\images\5.jpg)

 K-Max Pooling是一种全局取Top K特征的操作方式，而Chunk-Max Pooling则是先分段，在分段内包含特征数据里面取最大值，所以其实是一种局部Top K的特征抽取方式。

[自然语言处理中CNN模型几种常见的Max Pooling操作]

**Dynamic Pooling**

卷积时如果碰到triggle词，可以标记下不同色，max-pooling时按不同标记划分chunk。

动态k-max pooling层
k是动态的，计算出来的，即将卷积的输出结果在句长的维度上进行pooling，取出其最大的k个值：

![](\images\6.png)


其中k_top是指顶层所选取的k值，L表示总的卷积层数，l表示当前层数，s表示当前层在句子长度维度上的大小。

[CNN与句子分类之动态池化方法DCNN--模型介绍篇]

#### 模型结构的示例分析

分析一下《[A Sensitivity Analysis ](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1510.03820.pdf)...》

word embedding的维度是5，对于句子 i like this movie very muc，转换成矩阵AϵR7×5；
有6个卷积核，尺寸为(2×5), (3×5), (4×5)，每种尺寸各2个，A分别与以上卷积核进行卷积操作（这里的Stride Size相当于等于高度h）；

再用激活函数激活，每个卷积核得到了特征向量(feature maps)；
使用1-max pooling提取出每个feature map的最大值；

然后在级联得到最终的特征表达；
将特征输入至softmax layer进行分类, 在这层可以进行正则化操作( l2-regulariation)。

#### 实验参数分析

TextCNN模型中，超参数主要有词向量，Region Size的大小，Feature Map的数量，激活函数的选择，Pooling的方法，正则化的影响。《A Sensitivity Analysis...》论文前面几章对实验内容和结果进行了详细介绍，在9个数据集上基于Kim Y的模型做了大量的调参实验，得出AUC进行比较，根据的实验对比：

1）初始化词向量：一般不直接使用One-hot。除了随机初始化Embedding layer的外，使用预训练的word2vec、 GloVe初始化的效果都更加好（具体哪个更好依赖于任务本身）。非静态的比静态的效果好一些。

2）卷积核的尺寸filter_sizes：影响较大，通常过滤器的大小范围在1-10之间，一般取为3-5，对于句子较长的文本（100+），则应选择大一些。为了找到最优的过滤器大小(Filter Region Size)，可以使用线性搜索的方法。对不同尺寸ws的窗口进行结合会对结果产生影响。当把与最优ws相近的ws结合时会提升效果，但是如果将距离最优ws较远的ws相结合时会损害分类性能。刚开始，我们可以只用一个filter，调节Region Size来比对各自的效果，来看看那种size有最好的表现，然后在这个范围在调节不同Region的匹配。

3）卷积核的数量num_filters（对每个巻积核尺寸来说）：有较大的影响，一般取100~600（需要兼顾模型的训练效率） ，同时一般使用Dropout（0~0.5）。最好不要超过600，超过600可能会导致过拟合。可设为100-200。

4）激活函数：可以尽量多尝试激活函数，实验发现ReLU和tanh两种激活函数表现较佳。

5）池化选择：1-max pooling（1-max pooling的方式已经足够好了，相比于其他的pooling方式而言）。

6）Dropout和正则化：Dropout rate / dropout_keep_prob：dropout一般设为0.5。随着feature map数量增加，性能减少时，可以考虑增大正则化的力度，如尝试大于0.5的Dropout。

   正则化的作用微乎其微，正则项对最终模型性能的影响很小。l2正则化效益很小，所以这里建议设置一个比较大的L2 norm constrain，相比而言，dropout在神经网络中有着广泛的使用和很好的效果。

7）为了检验模型的性能水平，多次反复的交叉验证是必要的，这可以确保模型的高性能并不是偶然。

8） 随机性影响：由于模型训练过程中的随机性因素，如随机初始化的权重参数，mini-batch，随机梯度下降优化算法等，造成模型在数据集上的结果有一定的浮动，如准确率(accuracy)能达到1.5%的浮动，而AUC则有3.4%的浮动。

其它的训练参数：batch_size：64；num_epochs：10；每checkpoint_every：100轮便保存模型；仅保存最近num_checkpoints：5次模型

[论文笔记：A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification]

#### 其他相关模型

将卷积操作推广到句法依存树上。每个词都计算其依存树上的k个祖先，k不同即filter不同。即每个窗口围绕句法树中的一个结点，池化过程在不同的结点上进行。

