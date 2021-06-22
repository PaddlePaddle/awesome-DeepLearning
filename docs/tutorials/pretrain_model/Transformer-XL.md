# Transformer-XL： Attentive Language Models Beyonds a Fixed-Length Context
<br>

## 1. Transformer-XL的由来

在正式讨论Transformer-XL之前，我们先来看看经典的Transformer（后文称**Vanilla Transformer**）是如何处理数据和训练评估模型的将，如**图1**所示。

![image-20210604171637306](https://raw.githubusercontent.com/1649759610/images_for_blog/master/image-20210604171637306.png)

<center>图1 Vanilla Transformer 训练和评估阶段</center>

在**数据处理**方面，给定一串较长的文本串，**Vanilla Transformer**会按照固定的长度（比如512），直接将该文本串进行划分成若干Segment。这个处理方式不会关注文本串中语句本身的边界（比如标点或段落），这样"粗暴"的划分通常会将一句完整的话切分到两个Segment里面，导致上下文碎片化（**context fragmentation**）。另外，Transformer本身能够维持的依赖长度很有可能会超出这个固定的**划分长度**，从而导致Transformer能够捕获的最大依赖长度不超过这个划分长度，Transformer本身达不到更好的性能。

在**模型训练**方面，如**图1a**所示，**Vanilla Transformer**每次传给模型一个Segment进行训练，第1个Segment训练完成后，传入第2个Segment进行训练，然而前后的这两个Segment是没有任何联系的，也就是前后的训练是独立的。但事实是前后的Segment其实是有关联的。

在**模型评估**方面，如**图1b**所示，**Vanilla Transformer**会采用同训练阶段一致的**划分长度**，但仅仅预测最后一个位置的token，完成之后，整个序列向后移动一个位置，预测下一个token。这个处理方式保证了模型每次预测都能使用足够长的上下文信息，也缓解了训练过程中的context framentation问题。但是每次的Segment都会重新计算，计算代价很大。

基于上边的这些不足，**Transformer-XL**被提出来解决这些问题。它主要提出了两个技术：**Segment-Level 循环机制**和**相对位置编码**。**Transformer-XL**能够建模更长的序列依赖，比RNN长80%，比**Vanilla Transformer**长450%。同时具有更快的评估速度，比**Vanilla Transformer**快1800+倍。同时在多项任务上也达到了SoTA的效果。

## 2. Transformer-XL 建模更长序列

### 2.1 Segment-Level 循环机制

**Transformer-XL**通过引入**Segment-Level recurrence mechanism**来建模更长序列，它通过融合前后两个Segment的信息来到这个目的。

这里循环机制和RNN循环机制类似，在RNN中，每个时刻的RNN单元会接收上个时刻的输出和当前时刻的输入，然后将两者融合计算得出当前时刻的输出。**Transformer-XL**同样是接收上个时刻的输出和当前时刻的输入，然后将两者融合计算得出当前时刻的输出。但是两者的处理单位并不相同，RNN的处理单位是一个词，**Transformer-XL**的处理单位是一个Segment。**图2**展示了**Transformer-XL**在训练阶段和评估阶段的Segment处理方式。

![image-20210604181648404](https://raw.githubusercontent.com/1649759610/images_for_blog/master/image-20210604181648404.png)

<center>图2 Transformer-XL的训练和评估阶段</center>

在**模型训练**阶段，如**图2a**所示，**Transformer-XL**会缓存前一个Segment的输出序列，在计算下一个Segment的输出时会使用上一个Segment的缓存信息，将前后不同Segment的信息进行融合，能够帮助模型看见更远的地方，建模更长的序列依赖能力，同时也避免了context fragmentation问题。

在**模型评估**阶段，如**图2b**所示，**Transformer-XL**通过缓存前一个Segment的输出序列，当下一个Segment需要用这些输出时（前后两个Segment具有大部分的重复），不需要重新计算，从而加快了推理速度。

下边我们来具体聊聊这些事情是怎么做的，假设前后的两个Segment分别为：$\text{s}_{\tau}=[x_{\tau,1},x_{\tau,2},...,x_{\tau,L}]$和$\text{s}_{\tau+1}=[x_{\tau+1,1},x_{\tau+1,2},...,x_{\tau+1,L}]$，其中序列长度为$L$。另外假定$h_{\tau}^n \in \mathbb{R}^{L \times d}$为由$\text{s}_{\tau}$计算得出的第$n$层的状态向量，则下一个Segment $\text{s}_{\tau+1}$的第$n$层可按照如下方式计算：

$$
\begin{align}
& \tilde{h}_{\tau+1}^{n-1} = \left[ \text{SG}(h_{\tau}^{n-1}) \; \circ \;h_{\tau+1}^{n-1} \right] \\
& q_{\tau+1}^{n}, \; k_{\tau+1}^n, \; v_{\tau+1}^n = h_{\tau+1}^{n-1}W_{q}^{\mathrm{ T }}, \; \tilde{h}_{\tau+1}^{n-1}W_{k}^{\mathrm{ T }}, \; \tilde{h}_{\tau+1}^{n-1}W_{v}^{\mathrm{ T }} \\
& h_{n+1}^n = \text{Transformer-Layer}(q_{\tau+1}^{n}, \; k_{\tau+1}^n, \; v_{\tau+1}^n)
\end{align}
$$

其中，$\text{SG}(h_{\tau}^{n-1}) $表示不使用梯度，$\left[ \text{SG}(h_{\tau}^{n-1}) \; \circ \;h_{\tau+1}^{n-1} \right]$表示将前后两个Segment的输出向量在序列维度上进行拼接。中间的公式表示获取Self-Attention计算中相应的$q,k,v$矩阵，其中在计算$q$的时候仅仅使用了当前Segment的向量，在计算$k$和$v$的时候同时使用前一个Segment和当前Segment的信息。最后通过Self-Attention融合计算，得出当前Segment的输出向量序列。

### 2.2 相对位置编码

**Segment-Level recurrence mechanism**看起来已经做到了长序列建模，但是这里有个问题需要进一步讨论一下。我们知道，在**Vanilla Transformer**使用了绝对位置编码，我们来看看如果将绝对位置编码应用到$Segment-Level recurrence mechanism$中会怎样。

还是假设前后的两个Segment分别为：$\text{s}_{\tau}=[x_{\tau,1},x_{\tau,2},...,x_{\tau,L}]$和$\text{s}_{\tau+1}=[x_{\tau+1,1},x_{\tau+1,2},...,x_{\tau+1,L}]$，其中序列长度为$L$。每个Segment的Position Embedding矩阵为$U_{1:L} \in \mathbb{R}^{L \times d}$,  每个Segment $\text{s}_{\tau}$的词向量矩阵为$E_{\text{s}_{\tau}} \in \mathbb{R}^{L \times d}$，在**Vanilla Transformer**中，两者相加输入模型参与计算，如下式所示：

$$
h_{\tau+1} = f(h_{\tau},\; E_{\text{s}_{\tau+1}}+U_{1:L}) \\
h_{\tau} = f(h_{\tau-1},\; E_{\text{s}_{\tau}}+U_{1:L}) 
$$

很明显，如果按照这个方式计算，前后两个段$E_{\text{s}_{\tau}}$和$E_{\text{s}_{\tau+1}}$将具有相同的位置编码，这样两者信息融合的时候肯定会造成位置信息混乱。为了避免这份尴尬的操作，**Transformer-XL**使用了**相对位置编码**。

**相对位置**是通过计算两个token之间的距离定义的，例如第5个token相对第2个token之间的距离是3， 那么位置$i$相对位置$j$的距离是$i-j$，假设序列之中的最大相对距离$L_{max}$，则我们可以定义这样的一个相对位置矩阵$R \in \mathbb{R}^{L_{max} \times d}$，其中$R_k$表示两个token之间距离是$k$的相对位置编码向量。注意在**Transformer-XL**中，相对位置编码向量不是可训练的参数，以$R_k = [r_{k,1}, r_{k,2},...,r_{k,d}]$为例，每个元素通过如下形式生成：

$$
r_{b,2j} = \text{sin}(\frac{b}{10000^{2j/d}}), \quad r_{b,2j+1} = \text{cos}(\frac{b}{10000^{(2j)/d}})
$$

**Transformer-XL**将相对位置编码向量融入了Self-Attention机制的计算过程中，这里可能会有些复杂，我们先来看看**Vanilla Transformer**的Self-Attention计算过程，如下：

$$
\begin{align}
A_{i,j}^{\text{abs}} &= (W_q(E_{x_i}+U_i))^{\text{T}}(W_k(E_{x_j}+U_j))) \\
&= \underbrace {E_{x_i}^{\text{T}} W_q^{\text{T}} W_k E_{x_j}}_{(a)} + \underbrace {E_{x_i}^{\text{T}} W_q^{\text{T}} W_k U_j}_{(b)} + \underbrace {U_{i}^{\text{T}} W_q^{\text{T}} W_k E_{x_j}}_{(c)} + \underbrace {U_{i}^{\text{T}} W_q^{\text{T}} W_k U_{j}}_{(d)}
\end{align}
$$

其中$E_{x_i}$表示token $x_i$的词向量，$U_i$表示其绝对位置编码，根据这个展开公式，**Transformer-XL**将相对位置编码信息融入其中，如下：

$$
\begin{align}
A_{i,j}^{\text{rel}} = \underbrace {E_{x_i}^{\text{T}} W_q^{\text{T}} W_{k,E} E_{x_j}}_{(a)} + \underbrace {E_{x_i}^{\text{T}} W_q^{\text{T}} W_{k,R} R_{i-j}}_{(b)} + \underbrace {u^{\text{T}} W_{k,E} E_{x_j}}_{(c)} + \underbrace {v^{\text{T}} W_{k,R} R_{i-j}}_{(d)}
\end{align}
$$


这里做了这样几处改变以融入相对位置编码：

1. 在分项$(b)$和$(d)$中，使用相对位置编码$R_{i-j}$取代绝对位置编码$U_j$。
2. 在分项$(c)$和$(d)$中，使用可训练参数$u$和$v$取代$U_{i}^{\text{T}} W_q^{\text{T}}$。因为$U_{i}^{\text{T}} W_q^{\text{T}}$表示第$i$个位置的query 向量，这个query向量对于其他要进行Attention的位置来说都是一样的，因此可以直接使用统一的可训练参数进行替换。
3. 在所有分项中，使用$W_{k,E}$和$W_{k,R}$计算基于内容(词向量)的key向量和基于位置的key向量。

式子中的每个分项分别代表的含义如下：

1. $(a)$描述了基于内容的Attention
2. $(b)$描述了内容对于每个相对位置的bias
3. $(c)$描述了内容的全局bias
4. $(d)$描述了位置的全局bias

### 2.3 完整的Self-Attention计算过程

上边描述了**Transformer-XL**中的两个核心技术：**Segment-Level 循环机制**和**相对位置编码**，引入了这两项技术之后，**Transformer-XL**中从第$n-1$层到第$n$层完整的计算过程是这样的：

$$
\begin{align}
 \tilde{h}_{\tau}^{n-1} &= \left[ \text{SG}(h_{\tau-1}^{n-1}) \; \circ \;h_{\tau}^{n-1} \right] \\
 q_{\tau}^{n}, \; k_{\tau}^n, \; v_{\tau}^n &= h_{\tau}^{n-1}{W_{q}^n}^{\mathrm{ T }}, \; \tilde{h}_{\tau}^{n-1}{W_{k,E}^n}^{\mathrm{ T }}, \; \tilde{h}_{\tau}^{n-1}{W_{v}^n}^{\mathrm{ T }} \\
 A_{\tau,i,j}^{n} &= {q_{\tau, i}^{n}}^{\text{T}}k_{\tau,j}^{n} + {q_{\tau, i}^{n}}^{\text{T}}W_{k,R}^{n}R_{i-j} + u^{\text{T}}k_{\tau,j} + v^{\text{T}}W_{k,R}^{n}R_{i-j}  \\
{\alpha}_{\tau}^n &= \text{Masked-Softmax}(A_{\tau}^n)v_{\tau}^n \\
{\omicron}_{\tau}^n & = \text{LayerNorm}(\text{Linear}({\alpha}_{\tau}^n)+h_{\tau}^{n-1}) \\
h_{\tau}^n &= \text{Positionwise-Feed-Forward}({\omicron}_{\tau}^n)
\end{align}
$$

## 3. 相关资料

1. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf)
2. [Transformer-XL Github](https://github.com/kimiyoung/transformer-xl)
