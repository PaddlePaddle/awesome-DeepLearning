# 应用一：LSTM-深度语义文本匹配模型
## 1 LSTM的前向计算与反向传播
### 前向计算
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f71bd5390521413889620a3795661411d179c5e0500943e4b8c9f782ddd6d232" width="400" hegiht="" ></center>
<center><br>图1：LSTM门控单元</br></center>
<br></br>

- 输入门：$i_{t}=sigmoid(W_{i}X_{t}+V_{i}H_{t-1}+b_i)$，控制有多少输入信号会被融合。
- 遗忘门：$f_{t}=sigmoid(W_{f}X_{t}+V_{f}H_{t-1}+b_f)$，控制有多少过去的记忆会被遗忘。
- 输出门：$o_{t}=sigmoid(W_{o}X_{t}+V_{o}H_{t-1}+b_o)$，控制最终输出多少融合了记忆的信息。
- 单元状态：$g_{t}=tanh(W_{g}X_{t}+V_{g}H_{t-1}+b_g)$，输入信号和过去的输入信号做一个信息融合。

通过学习这些门的权重设置，长短时记忆网络可以根据当前的输入信号和记忆信息，有选择性地忽略或者强化当前的记忆或是输入信号，帮助网络更好地学习长句子的语义信息：

- 记忆信号：$c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot g_{t}$

- 输出信号：$h_{t} = o_{t} \cdot tanh(c_{t})$
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f1d63a14a81743f0bb501b1a433844739fce983febf84d68b724d9645ed9fd43" width="400" hegiht="" ></center>
<center><br>图2：LSTM前向计算</br></center>
<br></br>

### 反向传播
长短期记忆神经网络的训练算法同样采用反向传播算法，主要有以下三个步骤：
1. 前向计算每个神经元的输出值,对于长短期记忆神经网络来说，即 $f_{t} 、 i_{t}$ 、$c_{t}, O_{t}, h_{t}$ 五个向量的值；
1. 反向计算每个神经元的误差项 $\delta$ 值。与循环神经网络一样，长短期记忆神经网络误差项的反向传播也是包括两个方向: 一个是沿时间轴的反向传播，即从当前 $\boldsymbol{t}$时刻开始，计算每个时刻的误差项; 一个是延网络层的反向传播，误差项向上一层传播，计算每一层的误差项；
1. 根据每个时刻的误差项，计算每个权重参数的误差梯度, 更新权重参数。
具体的数学计算过程如下：
激活函数 $\sigma$ 和 $\tanh$ 导数形式为：
$$
\begin{gathered}
\sigma^{\prime}(x)=\frac{1}{1+e^{-x}}\left(1-\frac{1}{1+e^{-x}}\right)=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}} \\
\tanh ^{\prime}(x)=1-\left(\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}\right)^{2}
\end{gathered}
$$
设误差为 $L$，则在 $t$ 时刻的残差为:
$$
\delta_{t}=\frac{\partial L}{\partial h_{t}}
$$
定义 $\bar{f}_{t}, \overline{\mathbf{i}}, \overline{\tilde{\mathbf{c}}}_{t}, \overline{\mathbf{o}}_{t}$ 分别为:
$$
\begin{aligned}
&\bar{f}_{t}=w_{f} h_{t-1}+u_{f} x_{t}+b_{f} \\
&\bar{i}_{t}=w_{i} h_{t-1}+u_{i} x_{t}+b_{i} \\
&\overline{\tilde{c}}_{t}=w_{c} h_{t-1}+u_{c} x_{t}+b_{c} \\
&\bar{o}_{t}=w_{o} h_{t-1}+u_{o} x_{t}+b_{0}
\end{aligned}
$$
设 $\delta_{f, t}, \quad \delta_{\mathrm{i}, t}, \quad \delta_{\tilde{\mathfrak{c}}, t}, \quad \delta_{\mathrm{o}, t}$ 分别为:
$$
\begin{aligned}
\delta_{f, t} &=\frac{\partial L}{\partial \bar{f}_{t}} \\
\delta_{i, t} &=\frac{\partial L}{\partial \bar{i}_{t}} \\
\delta_{\tilde{c}, t} &=\frac{\partial L}{\partial \overline{\tilde{c}}_{t}} \\
\delta_{o, t} &=\frac{\partial L}{\partial \bar{o}_{t}}
\end{aligned}
$$
对于 $t-1$ 时刻的误差，则有：
$$
\delta_{t-1}^{T}=\frac{\partial L}{\partial h_{t-1}}=\frac{\partial L}{\partial h_{t}} \frac{\partial h_{t}}{\partial h_{t-1}}=\delta_{t}^{T} \frac{\partial h_{t}}{\partial h_{t-1}}
$$
根据全导数公式进一步可得：
$$
\delta_{t}^{T} \frac{\partial h_{t}}{\partial h_{t-1}}=\delta_{o, t}^{T} \frac{\partial \bar{o}_{t}}{\partial h_{t-1}}+\delta_{f, t}^{T} \frac{\partial \bar{f}_{t}}{\partial h_{t-1}}+\delta_{i, t}^{T} \frac{\partial_{t} \bar{i}_{t}}{\partial h_{t-1}}+\delta_{\tilde{\varepsilon}, t}^{T} \frac{\partial \overline{\tilde{c}}_{t}}{\partial h_{t-1}}
$$
接下来：
$$
\begin{aligned}
&\frac{\partial \bar{o}_{t}}{\partial h_{t-1}}=w_{o} \\
&\frac{\partial \bar{f}_{t}}{\partial h_{t-1}}=w_{f} \\
&\frac{\partial \overline{\dot{4}}}{\partial h_{t-1}}=w_{i} \\
&\frac{\partial \overline{\tilde{c}}_{t}}{\partial h_{t-1}}=w_{c}
\end{aligned}
$$
由此可得到 $\delta_{t-1}$，由此完成误差延时间轴的反向计算:
$$
\delta_{t-1}=\delta_{o, t}^{T} w_{o}+\delta_{f, t}^{T} w_{f}+\delta_{i, t}^{T} w_{i}+\delta_{\tilde{c}, t}^{T} w_{\tilde{c}}
$$
对于 $t$ 时刻的网络参数梯度, 设学习率为 $\eta$, 据此更新 LSTM 神经网络权重参数 $w$ :
$$
\begin{aligned}
&\frac{\partial L}{\partial w_{o}}=\frac{\partial L}{\partial \bar{o}_{t}} \frac{\partial \bar{o}_{t}}{\partial w_{o}}=\delta_{o, t} h_{t-1}^{T} \rightarrow w_{o}^{\prime}=w_{o}-\eta \delta_{o, t} h_{t-1}^{T} \\
&\frac{\partial L}{\partial w_{f}}=\frac{\partial L}{\partial \bar{f}_{t}} \frac{\partial \bar{f}_{t}}{\partial w_{f}}=\delta_{f, t} h_{t-1}^{T} \rightarrow w_{f}^{\prime}=w_{f}-\eta \delta_{f, t} h_{t-1}^{T} \\
&\frac{\partial L}{\partial w_{i}}=\frac{\partial L}{\partial \bar{i}} \frac{\partial \bar{i}}{\partial w_{i}}=\delta_{i, t} h_{t-1}^{T} \rightarrow w_{i}^{\prime}=w_{i}-\eta \delta_{i t} h_{t-1}^{T} \\
&\frac{\partial L}{\partial w_{c}}=\frac{\partial L}{\partial \bar{c}_{t}} \frac{\partial \overline{\tilde{c}}_{t}}{\partial w_{c}}=\delta_{\tilde{c}, t} h_{t-1}^{T} \rightarrow w_{c}^{\prime}=w_{c}-\eta \delta_{\varepsilon, t} h_{t-1}^{T}
\end{aligned}
$$
更新权重参数 $u$ :
$$
\begin{aligned}
&\frac{\partial L}{\partial u_{o}}=\frac{\partial L}{\partial \bar{o}_{t}} \frac{\partial \bar{o}_{t}}{\partial u_{o}}=\delta_{o, t} x_{t}^{T} \rightarrow u_{o}^{\prime}=u_{o}-\eta \delta_{o t} x_{t}^{T} \\
&\frac{\partial L}{\partial u_{f}}=\frac{\partial L}{\partial \bar{f}_{t}} \frac{\partial \overline{f_{t}}}{\partial u_{f}}=\delta_{f, t} x_{t}^{T} \rightarrow u_{f}^{\prime}=u_{f}-\eta \delta_{f, t} x_{t}^{T} \\
&\frac{\partial L}{\partial u_{i}}=\frac{\partial L}{\partial \bar{i}_{t}} \frac{\partial \bar{i}}{\partial u_{i}}=\delta_{i, t} x_{t}^{T} \rightarrow u_{i}^{\prime}=u_{i}-\eta \delta_{i, t} x_{t}^{T} \\
&\frac{\partial L}{\partial u_{c}}=\frac{\partial L}{\partial \bar{c}_{t}} \frac{\partial \overline{\tilde{c}}_{t}}{\partial u_{c}}=\delta_{\tilde{c}, t} x_{t}^{T} \rightarrow u_{\tilde{c}}^{\prime}=u_{\tilde{c}}-\eta \delta_{\tilde{c}, t} x_{t}^{T}
\end{aligned}
$$
更新权重参数 $b$ :
$$
\begin{aligned}
&\frac{\partial L}{\partial b_{o}}=\frac{\partial L}{\partial \bar{o}_{t}} \frac{\partial \bar{o}_{t}}{\partial b_{0}}=\delta_{o, t} \rightarrow b_{o}^{\prime}=b_{o}-\delta_{o, t} \\
&\frac{\partial L}{\partial b_{f}}=\frac{\partial L}{\partial \bar{f}_{t}} \frac{\partial \bar{f}_{t}}{\partial b_{f}}=\delta_{f, t} \rightarrow b_{f}^{\prime}=b_{f}-\delta_{f, t} \\
&\frac{\partial L}{\partial b_{i}}=\frac{\partial L}{\partial \bar{i}_{t}} \frac{\partial \bar{i}_{t}}{\partial b_{i}}=\delta_{i, t} \rightarrow b_{i}^{\prime}=b_{i}-\delta_{i, t} \\
&\frac{\partial L}{\partial b_{c}}=\frac{\partial L}{\partial \overline{\tilde{c}}_{t}} \frac{\partial \overline{\tilde{c}}_{t}}{\partial b_{c}}=\delta_{\tilde{c}, t} \rightarrow b_{\tilde{c}}^{\prime}=b_{\tilde{c}}-\delta_{c, t}
\end{aligned}
$$



## 2 LSTM - 深度语义文本匹配模型

针对深层语义特征表示问题，利用循环神经网络模型的结构优点提取深层语义特征，当处理序列变长时，循环神经网络的神经单元状态随着信息的不断叠加导致距离当前时刻较远的信息会遗失，也就是面临长期依赖的问题。

通过在循环神经网络单元中加入门控机制，可以存储较长语句序列的信息以保证原始语义的表达。

因此，提出使用长短期记忆网络来实现深层语义向量的编码，以此构建语句序列的匹配网络。

首先通过 word2vec 将语句转换成单词向量形式，再将其输入构建的长短期记忆神经网络中，得到语句的向量表示，然后再输入到一个多层感知机，最后通过输出层的 softmax 函数分类得到候选回复的概率值。基于长短期记忆网络的回复检索模型如图3所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/810ef7d9af694c74a138482813a6a48266b49ed65f954dc29c8ef18bdba9c81b" width="500" hegiht="" ></center>
<center><br>图3：深度语义文本匹配模型</br></center>
<br></br>





# 应用二：LSTM-基于文本摘要和情感挖掘的股票波动趋势预测

## 1 Seq2Seq - 文本摘要
### (1) Seq2Seq 模型 
自然语言处理中的大部分问题本质上都是序列化的，如词语构成句子，句子构成段落，而段落又组成文本，因此模型的输入和输出也都是序列化数据。Seq2Seq 模型包括 Encoder 和 Decoder 两个部分。首先将输入文本的向量序列 $x=\left(x_{1}, x_{2}, \cdots, x_{t}\right)$ 传到 Encoder 部分，在编码器部分逐个读取词 $x_{i}$，得到最后一个时间步长 $t$ 的固定维度向量 $c$，这就是 RNNcell 的基本功能。之后，保存了输入序列信息的 $c$ 被传入 Decoder 部分，同时还包括前一时刻的输出 $y_{t-1} 、$ 前一时刻的隐藏层 $h_{t-1}$, 即 $h_{t}=f\left(h_{t-1}, y_{t-1}, c\right)$ 。同样，根据 $h_{t}$, 能够求出第 $t$ 个词的 $y_{t}$ 的生成概率：$P\left(y_{t} \mid y_{(t-1)}, y_{t-2}, \cdots, y_{1}, c\right)=g\left(h_{t}, y_{t-1}, c\right)$ ，模型训练时则去最大化给定输入文本序列 $x$ 时输出文本序列为 $y$ 的条件概率:
$$
\max _{\theta} \frac{1}{N} \sum_{n=1}^{N} \log p_{\theta}\left(y_{n} \mid x_{n}\right)
$$
### (2)Attention 机制 
Seq2Seq 模型有效地建模了基于输入序列，预测未知输出序列的问题。但当输入句子过长时，模型性能会受到向量 $\mathrm{c}$ 的信息存储的影响，于是 Attention 注意力分配的机制被提出。 Attention 机制通过在每个时间输入不同的 $c$ 来解决这个问题。每一个 $c$ 会对输入序列所有隐藏层的信息 $\left(h_{1}, h_{2}, \cdots, h_{t}\right)$ 进行加权求和，选取与当前所需要输出词语 $y$ 最合适的上下文信息 $c_{t}$，即 $c_{t}=\sum_{j=1}^{T_{x}} \alpha_{i j} h_{t}$，其中 $\alpha_{t j}$ 代表权重。 Decoder在 $t$ 时刻的状态输出 $s_{t}$ 根据上一步的状态 $s_{t-1}, y_{t-1}, c_{t}$ 三者的一个非线性函数得出，如图1所示。相比较之前的 Encoder-Decoder 模型，Attention 机制最大的区别就在于它不再要求 Encoder 将所有的输入信息都编码进一个固定长度的向量 c 中，而是充分利用输入序列携带的信息，对输入的重要程度进行区分。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/819d611a086142adbabfe9b141b7505473c064b326874059a5b645670feff7c7" width="400" hegiht="" ></center>
<center><br>图4：基于Attention机制的Seq2Seq模型</br></center>
<br></br>

## 2 文本情感计算规则 
使用基于情感词典的语义规则来计算新闻文本的情感词：情感词是指能够对情感进行表达的一类词，通常情况下能够通过一个词来判断其想表达的情感态度，但是将情感词放入句子中和其他句子成分组合在一起，那么整句话的情感极性和程度可能会发生变化。尤其是当句子中有程度副词和否定副词来修饰情感词时，这些词的词性和位置的变化，都会对句子情感态度有较大的影响。因此，可以通过构造相关的语义规则来更深层次地挖掘情感词在各语境中的真实情感。根据情感词数量，将文本分为多个族，文本的情感值为族的情感值的和。计算公式如下:
$$
\begin{gathered}
U_{1}=\prod L_{s} \times S \\
U_{2}=\left(\prod L_{n} \times\left(0.1 \times C_{n}-1\right)^{c_{n}}+\prod L_{s}\right) \times S
\end{gathered}
$$
其中, $U_{1}$ 为无否定词的族的情感值；$U_{2}$ 为有否定词的族的情感值；$S$ 为情感词的情感值；$L_{s}$ 为修饰情感词的程度副词的程度值；$L_{n}$ 为修饰否定词的程度副词的程度值；$C_{n}$ 为否定词的数量。

## 3 LSTM - 股票预测模型
LSTM $^{[32]}$ (long short-term memory) 是一种特殊的循环神经网络( recurrent nerual network, RNN)，通过良好的设计来记住长期的信息，进而避免长期依赖(long-term dependency) 问题。在进行股票波动趋势预测过程中，下一天的预测值往往基于其股票历史数据，而 LSTM 可以直接对任意长度的序列进行处理，能够满足股票预测需求。

LSTM 通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态( cell state) 的能力。门是一种让信息选择式通过的方法，其中包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。LSTM 的门分为遗忘门 $\left(f_{t}\right)$ 输入门 $\left(i_{t}\right)$ 和输出门 $\left(o_{t}\right)$ 。对输入的股票序列 $x=\left\{x_{1}, x_{2}, \cdots, x_{m}\right\}$, 忘记门会读取 $h_{t-1}$ 和 $x_{t}$ 决定从细胞状态中丢弃的信息，即：
$$
f_{t}=\operatorname{sigmoid}\left(W_{f}\left[h_{t-1}, x_{t}\right]+b_{f}\right)
$$
之后，输入门确定要存储在细胞状态的新信息，即:
$$
i_{t}=\operatorname{sigmoid}\left(W_{i}\left[h_{t-1}, x_{t}\right]+b_{i}\right)
$$
随后，$\tanh$ 层创建一个新的候选值向量 $\tilde{c}_{1}, \tilde{c}_{t}=\tanh \left(W_{c}\left[h_{t-1}, x_{t}\right]+b_{i}\right)$，该值会被加入到状态中。接着更
新细胞状态，将旧状态 $c_{t-1}$ 与 $f_{t}$ 相乘，丢弃需要丢弃的信息，再加上输出门需要存储的候选值，即 $c_{t}=f_{t} \times c_{t-1}+i_{t} \times$
$\tilde{c}_{t}$ 。最终靠输出门输出细胞状态，即:
$$
\begin{gathered}
o_{t}=\operatorname{sigmoid}\left(W_{o}\left[h_{t-1}, x_{t}\right]+b_{o}\right) \\
h_{t}=o_{t} \times \tanh \left(c_{t}\right)
\end{gathered}
$$
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c95949227bc94716a41b76360b0e93d2100cfa993d8447c3be5672cff8a84b72" width="800" hegiht="" ></center>
<center><br>图5：NLP股票趋势预测模型设计结构图</br></center>
<br></br>
