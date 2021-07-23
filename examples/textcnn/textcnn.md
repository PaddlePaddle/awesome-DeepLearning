# textcnn

* 前言
* textcnn结构

## 前言

​	对于文本分类问题，常见的方法无非就是抽取文本的特征，比如使用doc2evc或者LDA模型将文本转换成一个固定维度的特征向量，然后在基于抽取的特征训练一个分类器。 然而研究证明，TextCnn在文本分类问题上有着更加卓越的表现。从直观上理解，TextCNN通过一维卷积来获取句子中n-gram的特征表示。TextCNN对文本浅层特征的抽取能力很强，在短文本领域如搜索、对话领域专注于意图分类时效果很好，应用广泛，且速度快，一般是首选；对长文本领域，TextCNN主要靠filter窗口抽取特征，在长距离建模方面能力受限，且对语序不敏感。

​	CNN可以识别出当前任务中具有预言性的n元语法（且如果使用特征哈希可以使用无约束的n元语法词汇，同时保持词嵌入矩阵的约束）；CNN卷积结构还允许有相似成分的n元语法分享预测行为，即使在预测过程中遇见未登录的特定n元语法；层次化的CNN每层有效着眼于句子中更长的n元语法，使得模型还可以对非连续n元语法敏感。

## textcnn结构

​	有几篇文章都是textcnn，模型结构类似。其中《[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)》给出了基本结构，《[A Sensitivity Analysis ](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1510.03820.pdf)...》专门做了各种控制变量的实验对比。

模型示意图：

![cnn](C:\Users\SongWood\shujiaxuexi\baidushixi\textcnn\images\cnn.png)

模型示意图2：

![textcnn](C:\Users\SongWood\shujiaxuexi\baidushixi\textcnn\images\textcnn.png)

### 1.嵌入层(embedding layer)

​	textcnn使用预先训练好的词向量作embedding layer。对于数据集里的所有词，因为每个词都可以表征成一个向量，因此我们可以得到一个嵌入矩阵MM, MM里的每一行都是词向量。这个MM可以是静态(static)的，也就是固定不变。可以是非静态(non-static)的，也就是可以根据反向传播更新。

### 2.卷积池化层(convolution and pooling)

#### 卷积

输入一个句子，首先对这个句子进行切词，假设有s个单词。对每个词，跟句嵌入矩阵M, 可以得到词向量。假设词向量一共有d维。那么对于这个句子，便可以得到s行d列的矩阵AϵRs×d. 
        我们可以把矩阵A看成是一幅图像，使用卷积神经网络去提取特征。由于句子中相邻的单词关联性总是很高的，因此可以使用一维卷积，即文本卷积与图像卷积的不同之处在于只在文本序列的一个方向（垂直）做卷积，卷积核的宽度固定为词向量的维度d。高度是超参数，可以设置。 对句子单词每个可能的窗口做卷积操作得到特征图(feature map) c = [c_1, c_2, …, c_s-h+1]。

​	现在假设有一个卷积核，是一个宽度为d，高度为h的矩阵w，那么w有h∗d个参数需要被更新。对于一个句子，经过嵌入层之后可以得到矩阵AϵRs×d。 A[i:j]表示A的第i行到第j行, 那么卷积操作可以用如下公式表示：

![公式](C:\Users\SongWood\shujiaxuexi\baidushixi\textcnn\images\公式.png)

​	对一个卷积核，可以得到特征cϵRs−h+1, 总共s−h+1个特征。我们可以使用更多高度h不同的卷积核，得到更丰富的特征表达。

​	TextCNN网络包括很多不同窗口大小的**卷积核**，常用的filter size ∈ {3,4,5}，每个filter的feature maps=100。这里的特征图就是不同的k元语法。如上图中分别有两个不同的二、三和四元语法。

​	如果设置padding='same'即使用宽卷积，则每个feature maps for each region size都是seq_len*1，所有的feature map可以拼接成seq_len*(num_filters*num_filter_size)的矩阵，回到输入类似维度，这样就可以使用多层cnn了。

​	图像中可以利用 (R, G, B) 作为不同channel。而文本的输入的channel通常是不同方式的embedding方式（比如 word2vec或Glove），实践中也有利用静态词向量和fine-tunning词向量作为不同channel的做法；channel也可以一个是词序列，另一个channel是对应的词性序列。接下来就可以通过加和或者拼接进行结合。

#### 池化

​	不同尺寸的卷积核得到的特征(feature map)大小也是不一样的，因此我们对每个feature map使用池化函数，使它们的维度相同。

​	最常用的就是1-max pooling，提取出feature map照片那个的最大值，通过选择每个feature map的最大值，可捕获其最重要的特征。这样每一个卷积核得到特征就是一个值，对所有卷积核使用1-max pooling，再级联起来，可以得到最终的特征向量，这个特征向量再输入softmax layer做分类。这个地方可以使用drop out防止过拟合。

​	CNN中采用Max Pooling操作有几个好处：首先，这个操作可以保证特征的位置与旋转不变性，因为不论这个强特征在哪个位置出现，都会不考虑其出现位置而能把它提出来。但是对于NLP来说，这个特性其实并不一定是好事，因为在很多NLP的应用场合，特征的出现位置信息是很重要的，比如主语出现位置一般在句子头，宾语一般出现在句子尾等等。     其次，MaxPooling能减少模型参数数量，有利于减少模型过拟合问题。因为经过Pooling操作后，往往把2D或者1D的数组转换为单一数值，这样对于后续的Convolution层或者全联接隐层来说无疑单个Filter的参数或者隐层神经元个数就减少了。        再者，对于NLP任务来说，可以把变长的输入X整理成固定长度的输入。因为CNN最后往往会接全联接层，而其神经元个数是需要事先定好的，如果输入是不定长的那么很难设计网络结构。

### 3.模型结构示例分析

​	word embedding的维度是5，对于句子 i like this movie very muc，转换成矩阵AϵR7×5；
有6个卷积核，尺寸为(2×5), (3×5), (4×5)，每种尺寸各2个，A分别与以上卷积核进行卷积操作（这里的Stride Size相当于等于高度h）；

​	再用激活函数激活，每个卷积核得到了特征向量(feature maps)；使用1-max pooling提取出每个feature map的最大值；

​	然后在级联得到最终的特征表达；将特征输入至softmax layer进行分类, 在这层可以进行正则化操作( l2-regulariation)。

​	TextCNN模型中，超参数主要有词向量，Region Size的大小，Feature Map的数量，激活函数的选择，Pooling的方法，正则化的影响。《A Sensitivity Analysis...》论文前面几章对实验内容和结果进行了详细介绍，在9个数据集上基于Kim Y的模型做了大量的调参实验，得出AUC进行比较，根据的实验对比：

1）初始化词向量：一般不直接使用One-hot。除了随机初始化Embedding layer的外，使用预训练的word2vec、 GloVe初始化的效果都更加好（具体哪个更好依赖于任务本身）。非静态的比静态的效果好一些。

2）卷积核的尺寸filter_sizes：影响较大，通常过滤器的大小范围在1-10之间，一般取为3-5，对于句子较长的文本（100+），则应选择大一些。为了找到最优的过滤器大小(Filter Region Size)，可以使用线性搜索的方法。对不同尺寸ws的窗口进行结合会对结果产生影响。当把与最优ws相近的ws结合时会提升效果，但是如果将距离最优ws较远的ws相结合时会损害分类性能。刚开始，我们可以只用一个filter，调节Region Size来比对各自的效果，来看看那种size有最好的表现，然后在这个范围在调节不同Region的匹配。

3）卷积核的数量num_filters（对每个巻积核尺寸来说）：有较大的影响，一般取100~600（需要兼顾模型的训练效率） ，同时一般使用Dropout（0~0.5）。最好不要超过600，超过600可能会导致过拟合。可设为100-200。

4）激活函数：可以尽量多尝试激活函数，实验发现ReLU和tanh两种激活函数表现较佳。

5）池化选择：1-max pooling（1-max pooling的方式已经足够好了，相比于其他的pooling方式而言）。

6）Dropout和正则化：Dropout rate / dropout_keep_prob：dropout一般设为0.5。随着feature map数量增加，性能减少时，可以考虑增大正则化的力度，如尝试大于0.5的Dropout。

   正则化的作用微乎其微，正则项对最终模型性能的影响很小。l2正则化效益很小，所以这里建议设置一个比较大的L2 norm constrain，相比而言，dropout在神经网络中有着广泛的使用和很好的效果。

7）为了检验模型的性能水平，多次反复的交叉验证是必要的，这可以确保模型的高性能并不是偶然。

8） 随机性影响：由于模型训练过程中的随机性因素，如随机初始化的权重参数，mini-batch，随机梯度下降优化算法等，造成模型在数据集上的结果有一定的浮动，如准确率(accuracy)能达到1.5%的浮动，而AUC则有3.4%的浮动。

