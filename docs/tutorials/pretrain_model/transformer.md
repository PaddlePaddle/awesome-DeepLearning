# Transformer

## 1.介绍

Transformer 网络架构架构由 Ashish Vaswani 等人在 Attention Is All You Need一文中提出，并用于机器翻译任务，和以往网络架构有所区别的是，该网络架构中，编码器和解码器没有采用 RNN 或 CNN 等网络架构，而是采用完全依赖于注意力机制的架构。网络架构如下所示：

![](../../images/pretrain_model/Transformer/transformer.png)

Transformer改进了RNN被人诟病的训练慢的特点，利用self-attention可以实现快速并行。下面的章节会详细介绍Transformer的各个组成部分。

## 2.Transformer直观认识
Transformer主要由encoder和decoder两部分组成。在Transformer的论文中，encoder和decoder均由6个encoder layer和decoder layer组成，通常我们称之为encoder block。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/739aa8a15ec043f8920843fa142754276c10b5b082d64b39a7d0f795241ace82"  width="600px" /></center> 
<center><br>transformer结构 </br></center>
<br></br>

每一个encoder和decoder的内部简版结构如下图

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/175551e7b2f44de6948732de030723aa258fc317dd78472dbcb2d25570b92c9d"  width="600px" /></center> 
<center><br>transformer的encoder或者decoder的内部结构 </br></center>
<br></br>

对于encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。

decoder也包含encoder提到的两层网络，但是在这两层中间还有一层attention层，帮助当前节点获取到当前需要关注的重点内容。



首先，模型需要对输入的数据进行一个embedding操作，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/fa7dd09419ca4b9d972aa3b8f99926ccc41a60588cc34ae5aa730ce89e267cf0"  width="600px" /></center> 
<center><br>embedding和self-attention </br></center>
<br></br>

## 3. Transformer的结构
Transformer的结构解析出来如下图表示，包括Input Embedding, Position Embedding, Encoder, Decoder。

![](../../images/pretrain_model/Transformer/Transformer_architecture.png)

## 3.1 Embedding

![](../../images/pretrain_model/Transformer/input_embedding.png)

字向量与位置编码的公式表示如下：

$$X=Embedding Lookup(X)+Position Encoding$$

### 3.1.1 Input Embedding

可以将Input Embedding看作是一个 lookup table，对于每个 word，进行 word embedding 就相当于一个lookup操作，查出一个对应结果。


### 3.1.2 Position Encoding

Transformer模型中还缺少一种解释输入序列中单词顺序的方法。为了处理这个问题，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下

$$PE(pos,2i)=sin(pos/10000^{2i}/d_{model})$$
$$PE(pos,2i+1)=cos(pos/10000^{2i}/d_{model})$$

其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码.

## 3.2 Encoder

![](../../images/pretrain_model/Transformer/encoder.png)

用公式把一个Transformer Encoder block 的计算过程整理一下

+ 自注意力机制

$$Q=XW_{Q}$$
$$K=XW_{K}$$
$$V=XW_{V}$$

$$X_{attention}=selfAttention(Q,K,V)$$

+ self-attention 残差连接与 Layer Normalization


$$X_{attention}=LayerNorm(X_{attention})$$

+  FeedForward，其实就是两层线性映射并用激活函数激活，比如说RELU

$$X_{hidden}=Linear(RELU(Linear(X_{attention})))$$

+ FeedForward 残差连接与 Layer Normalization

$$X_{hidden}=X_{attention}+X_{hidden}$$

$$X_{hidden}=LayerNorm(X_{hidden})$$

其中：$X_{hidden} \in R^{batch_size*seq_len*embed_dim}$

### 3.2.1 自注意力机制
- 首先，自注意力机制（self-attention）会计算出三个新的向量，在论文中，向量的维度是512维，我们把这三个向量分别称为Query、Key、Value，这三个向量是用embedding向量与一个矩阵相乘得到的结果，这个矩阵是随机初始化的，维度为（64，512）注意第二个维度需要和embedding的维度一样，其值在反向传播的过程中会一直进行更新，得到的这三个向量的维度是64低于embedding维度的。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8f322d81d9c3491d9b714e39986e482926755e9c86dd4266805cceb4add145c7"  width="600px" /></center> 
<center><br>Query Key Value </br></center>
<br></br>

2、计算self-attention的分数值，该分数值决定了当我们在某个位置encode一个词时，对输入句子的其他部分的关注程度。这个分数值的计算方法是Query与Key做点乘，以下图为例，首先我们需要针对Thinking这个词，计算出其他词对于该词的一个分数值，首先是针对于自己本身即q1·k1，然后是针对于第二个词即q1·k2

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/04a8c9a33598465c80e63736e54c70b2c480a4feec9141dcb3487cf5a1f90f7a"  width="600px" /></center> 
<center><br>Query Key Value</br></center>
<br></br>

3、接下来，把点乘的结果除以一个常数，这里我们除以8，这个值一般是采用上文提到的矩阵的第一个维度的开方即64的开方8，当然也可以选择其他的值，然后把得到的结果做一个softmax的计算。得到的结果即是每个词对于当前位置的词的相关性大小，当然，当前位置的词相关性肯定会会很大

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ed1176ae195145fa8abf6635d4d6aaeb938d2d803b6d458b9cc41a9ea7e1914f"  width="600px" /></center> 
<center><br>softmax </br></center>
<br></br>

4、下一步就是把Value和softmax得到的值进行相乘，并相加，得到的结果即是self-attetion在当前节点的值。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/5be4009ebf3f43ce9a9947785bb6058ba7289b9adf9b4aba8f6682d949cd4b4f"  width="600px" /></center> 
<center><br>dot product </br></center>
<br></br>

在实际的应用场景，为了提高计算速度，我们采用的是矩阵的方式，直接计算出Query, Key, Value的矩阵，然后把embedding的值与三个矩阵直接相乘，把得到的新矩阵Q与K相乘，乘以一个常数，做softmax操作，最后乘上V矩阵

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/82e73917080f4f64917ae7a45bfa39596c2229f3dd0e4e489ff76e78fa627c93"  width="600px" /></center> 
<center><br> scaled dot product attention </br></center>
<br></br>

这种通过 query 和 key 的相似性程度来确定 value 的权重分布的方法被称为scaled dot-product attention。

用公式表达如下：

$$Q=XW_{Q}$$
$$K=XW_{K}$$
$$V=XW_{V}$$

$$X_{attention}=selfAttention(Q,K,V)$$


### 3.2.2  Self-Attention 复杂度

Self-Attention时间复杂度：$O(n^2 \cdot d)$ ，这里，n是序列的长度，d是embedding的维度。

Self-Attention包括三个步骤：相似度计算，softmax和加权平均，它们分别的时间复杂度是：

相似度计算可以看作大小为(n,d)和(d,n)的两个矩阵相乘： $(n,d) *(d,n) =(n^2 \cdot d)$，得到一个(n,n)的矩阵

softmax就是直接计算了，时间复杂度为: $O(n^2)$

加权平均可以看作大小为(n,n)和(n,d)的两个矩阵相乘：  $(n,d) *(d,n) =(n^2 \cdot d)$，得到一个(n,d)的矩阵

因此，Self-Attention的时间复杂度是: $O(n^2 \cdot d)$

### 3.2.3 Multi-head Attention

不仅仅只初始化一组Q、K、V的矩阵，而是初始化多组，tranformer是使用了8组，所以最后得到的结果是8个矩阵。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/11e365dd4a4145b79e6258c0d77ec77cac1a1f9a5dab4b31830459b2b7cb347a"  width="600px" /></center> 
<center><br> multi-head attention </br></center>
<br></br>

multi-head注意力的全过程如下，首先输入句子，“Thinking Machines”,在embedding模块把句子中的每个单词变成向量X，在encoder层中，除了第0层有embedding操作外，其他的层没有embedding操作；接着把X分成8个head，
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/4d634e35dea1472d9d6946b75c14f121fd274ec19e474f86bc4a903620d87f65"  width="600px" /></center> 
<center><br> multi-head attention总体结构 </br></center>
<br></br>

### 3.2.4  Multi-Head Attention复杂度

多头的实现不是循环的计算每个头，而是通过 transposes and reshapes，用矩阵乘法来完成的。

Transformer/BERT中把 d ，也就是hidden_size/embedding_size这个维度做了reshape拆分。并将 num_attention_heads 维度transpose到前面，使得Q和K的维度都是(m,n,a)，这里不考虑batch维度。

这样点积可以看作大小为(m,n,a)和(m,a,n)的两个张量相乘，得到一个(m,n,n)的矩阵，其实就相当于(n,a)和(a,n)的两个矩阵相乘，做了m次，时间复杂度是：

$$O(n^2 \cdot m \cdot a)=O(n^2 \cdot d)$$

因此Multi-Head Attention时间复杂度也是$O(n^2 \cdot d)$，复杂度相较单头并没有变化，主要还是transposes and reshapes 的操作，相当于把一个大矩阵相乘变成了多个小矩阵的相乘。



### 3.2.5 残差连接
经过 self-attention 加权之后输出，也就是Attention(Q,K,V) ，然后把他们加起来做残差连接

$$X_{hidden}=X_{embedding}+self Attention(Q,K,V)$$

除了self-attention这里做残差连接外，feed forward那个地方也需要残差连接，公式类似：

$$X_{hidden}=X_{feed_forward}+X_{hidden}$$


### 3.2.6 Layer Normalization

Layer Normalization 的作用是把神经网络中隐藏层归一为标准正态分布，也就是独立同分布，以起到加快训练速度，加速收敛的作用
$$X_{hidden}=LayerNorm(X_{hidden})$$

其中：$X_{hidden} \in R^{batch_size*seq_len*embed_dim}$

LayerNorm的详细操作如下：

$$\mu_{L}=\dfrac{1}{m}\sum_{i=1}^{m}x_{i}$$

上式以矩阵为单位求均值；

$$\delta^{2}=\dfrac{1}{m}\sum_{i=1}^{m}(x_{i}-\mu)^2$$
上式以矩阵为单位求方差

$$ LN(x_{i})=\alpha \dfrac{x_{i}-\mu_{L}}{\sqrt{\delta^{2}+\epsilon}}+\beta $$
然后用每一列的每一个元素减去这列的均值，再除以这列的标准差，从而得到归一化后的数值，加$\epsilon$是为了防止分母为0.此处一般初始化$\alpha$为全1，而$\beta$为全0.

### 3.2.7 Feed Forward


![](../../images/pretrain_model/Transformer/feed_forward.png)

将Multi-Head Attention得到的向量再投影到一个更大的空间（论文里将空间放大了4倍）在那个大空间里可以更方便地提取需要的信息（使用Relu激活函数），最后再投影回token向量原来的空间

$$FFN(x)=ReLU(W_{1}x+b_{1})W_{2}+b_{2}$$

借鉴SVM来理解：SVM对于比较复杂的问题通过将特征其投影到更高维的空间使得问题简单到一个超平面就能解决。这里token向量里的信息通过Feed Forward Layer被投影到更高维的空间，在高维空间里向量的各类信息彼此之间更容易区别。



## 3.3 Decoder

![](../../images/pretrain_model/Transformer/decoder.png)

和 Encoder 一样，上面三个部分的每一个部分，都有一个残差连接，后接一个 Layer Normalization。Decoder 的中间部件并不复杂，大部分在前面 Encoder 里我们已经介绍过了，但是 Decoder 由于其特殊的功能，因此在训练时会涉及到一些细节，下面会介绍Decoder的Masked Self-Attention和Encoder-Decoder Attention两部分，其结构图如下图所示

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/08f1c94cb40f4d76aba52e7d026897177f7b6f5f69804560bbed9e576094679f"  width="600px" /></center> 
<center><br> decoder self attention </br></center>
<br></br>


### 3.3.1 Masked Self-Attention


传统 Seq2Seq 中 Decoder 使用的是 RNN 模型，因此在训练过程中输入因此在训练过程中输入t时刻的词，模型无论如何也看不到未来时刻的词，因为循环神经网络是时间驱动的，只有当t时刻运算结束了，才能看到t+1时刻的词。而 Transformer Decoder 抛弃了 RNN，改为 Self-Attention，由此就产生了一个问题，在训练过程中，整个 ground truth 都暴露在 Decoder 中，这显然是不对的，我们需要对 Decoder 的输入进行一些处理，该处理被称为 Mask。

Mask 非常简单，首先生成一个下三角全 0，上三角全为负无穷的矩阵，然后将其与 Scaled Scores 相加即可，之后再做 softmax，就能将 -inf 变为 0，得到的这个矩阵即为每个字之间的权重。

### 3.3.2 Masked Encoder-Decoder Attention
其实这一部分的计算流程和前面 Masked Self-Attention 很相似，结构也一摸一样，唯一不同的是这里的K,V为 Encoder 的输出，Q为 Decoder 中 Masked Self-Attention 的输出

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/eeb0db3260814772afdc9c1566a4afaa7133168766f34670bdecb26875edcd3f"  width="600px" /></center> 
<center><br> Masked Encoder-Decoder Attention </br></center>
<br></br>

### 3.3.3 Decoder的解码

下图展示了Decoder的解码过程，Decoder中的字符预测完之后，会当成输入预测下一个字符，知道遇见终止符号为止。

![](../../images/pretrain_model/Transformer/transformer_decoding_2.gif)

## 3.4 Transformer的最后一层和Softmax

线性层是一个简单的全连接的神经网络，它将解码器堆栈生成的向量投影到一个更大的向量，称为logits向量。如图linear的输出

softmax层将这些分数转换为概率（全部为正值，总和为1.0）。选择概率最高的单元，并生成与其关联的单词作为此时间步的输出。如图softmax的输出。

![](../../images/pretrain_model/Transformer/linear_softmax.png)


## 3.5 Transformer的权重共享

Transformer在两个地方进行了权重共享：

+ （1）Encoder和Decoder间的Embedding层权重共享；

《Attention is all you need》中Transformer被应用在机器翻译任务中，源语言和目标语言是不一样的，但它们可以共用一张大词表，对于两种语言中共同出现的词（比如：数字，标点等等）可以得到更好的表示，而且对于Encoder和Decoder，嵌入时都只有对应语言的embedding会被激活，因此是可以共用一张词表做权重共享的。

论文中，Transformer词表用了bpe来处理，所以最小的单元是subword。英语和德语同属日耳曼语族，有很多相同的subword，可以共享类似的语义。而像中英这样相差较大的语系，语义共享作用可能不会很大。

但是，共用词表会使得词表数量增大，增加softmax的计算时间，因此实际使用中是否共享可能要根据情况权衡。

+ （2）Decoder中Embedding层和FC层权重共享；

Embedding层可以说是通过onehot去取到对应的embedding向量，FC层可以说是相反的，通过向量（定义为 x）去得到它可能是某个词的softmax概率，取概率最大（贪婪情况下）的作为预测值。

那哪一个会是概率最大的呢？在FC层的每一行量级相同的前提下，理论上和 x 相同的那一行对应的点积和softmax概率会是最大的（可类比本文问题1）。

因此，Embedding层和FC层权重共享，Embedding层中和向量 x 最接近的那一行对应的词，会获得更大的预测概率。实际上，Decoder中的Embedding层和FC层有点像互为逆过程。

通过这样的权重共享可以减少参数的数量，加快收敛。

## 4 总结

本文详细介绍了Transformer的细节，包括Encoder，Decoder部分，输出解码的部分，Transformer的共享机制等等。

## 5. 参考文献
[Attention Is All You Need](https://arxiv.org/abs/1706.03762)



