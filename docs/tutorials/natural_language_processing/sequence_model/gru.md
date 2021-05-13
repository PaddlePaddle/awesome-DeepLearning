# GRU

在神经网络发展的过程中，几乎所有关于LSTM的文章中对于LSTM的结构都会做出一些变动，也称为LSTM的变体。其中变动较大的是门控循环单元（Gated Recurrent Units），也就是较为流行的GRU。GRU是2014年由Cho, et al在文章《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中提出的，某种程度上GRU也是对于LSTM结构复杂性的优化。LSTM能够解决循环神经网络因长期依赖带来的梯度消失和梯度爆炸问题，但是LSTM有三个不同的门，参数较多，训练起来比较困难。GRU只含有两个门控结构，且在超参数全部调优的情况下，二者性能相当，但是GRU结构更为简单，训练样本较少，易实现。

![gru](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru.png)

GRU在LSTM的基础上主要做出了两点改变 ：

（1）GRU只有两个门。GRU将LSTM中的输入门和遗忘门合二为一，称为更新门（update gate），上图中的$z_{t}$，控制前边记忆信息能够继续保留到当前时刻的数据量，或者说决定有多少前一时间步的信息和当前时间步的信息要被继续传递到未来；GRU的另一个门称为重置门（reset gate），上图中的$r_{t}$ ，控制要遗忘多少过去的信息。

（2）取消进行线性自更新的记忆单元（memory cell），而是直接在隐藏单元中利用门控直接进行线性自更新。GRU的逻辑图如上图所示。

GRU的公式化表达如下：

$$z_{t}=\sigma(W_{z} \cdot [h_{t-1},x_{t}])$$
$$r_{t}=\sigma(W_{r} \cdot [h_{t-1},x_{t}])$$
$$\tilde h=tanh(W \cdot [r_{t} \odot h_{t-1},x_{t}])$$
$$h_{t}=(1-z_{t})\odot h_{t-1}+z_{t} \odot \tilde h_{t-1}$$

下面我们将分步介绍GRU的单元传递过程，公式也会在接下来的章节进行详细的介绍：

![gru](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru_1.png)

上图是带有门控循环单元的循环神经网络。

## 1.更新门

在时间步 t，我们首先需要使用以下公式计算更新门
$$z_{t}=\sigma(W_{z} \cdot [h_{t-1},x_{t}])$$

其中 $x_{t}$ 为第 t 个时间步的输入向量，即输入序列 X 的第 t 个分量，它会经过一个线性变换（与权重矩阵$W_{z}$ 相乘）。$h_{t-1}$保存的是前一个时间步 t-1 的信息，它同样也会经过一个线性变换。更新门将这两部分信息相加并投入到 Sigmoid 激活函数中，因此将激活结果压缩到 0 到 1 之间。以下是更新门在整个单元的位置与表示方法。

![gru](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru_zt.png)

更新门帮助模型决定到底要将多少过去的信息传递到未来，或到底前一时间步和当前时间步的信息有多少是需要继续传递的。这一点非常强大，因为模型能决定从过去复制所有的信息以减少梯度消失的风险。

## 2.重置门

本质上来说，重置门主要决定了到底有多少过去的信息需要遗忘，我们可以使用以下表达式计算：
$$r_{t}=\sigma(W_{r} \cdot [h_{t-1},x_{t}])$$
该表达式与更新门的表达式是一样的，只不过线性变换的参数和用处不一样而已。下图展示了该运算过程的表示方法。

![gru_rt](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru_rt.png)

如前面更新门所述，$h_{t-1}$和 $x_{t}$先经过一个线性变换，再相加投入 Sigmoid 激活函数以输出激活值。

## 3. 当前记忆内容

现在我们具体讨论一下这些门控到底如何影响最终的输出。在重置门的使用中，新的记忆内容将使用重置门储存过去相关的信息，它的计算表达式为：
$$\tilde h=tanh(W \cdot [r_{t} \odot h_{t-1},x_{t}])$$

输入$x_{t}$与上一时间步信息 $h_{t-1}$先经过一个线性变换，即分别右乘矩阵 W。

计算重置门 $r_{t}$与$h_{t-1}$ 的 Hadamard 乘积，即 $r_{t}$ 与$h_{t-1}$的对应元素乘积。因为前面计算的重置门是一个由 0 到 1 组成的向量，它会衡量门控开启的大小。例如某个元素对应的门控值为 0，那么它就代表这个元素的信息完全被遗忘掉。该 Hadamard 乘积将确定所要保留与遗忘的以前信息。

将这两部分的计算结果相加再投入双曲正切激活函数中。该计算过程可表示为：

![gru_memory](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru_memory.png)

## 4. 当前时间步的最终记忆

在最后一步，网络需要计算 $h_t$，该向量将保留当前单元的信息并传递到下一个单元中。在这个过程中，我们需要使用更新门，它决定了当前记忆内容 $\tilde h$和前一时间步 $h_{t-1}$中需要收集的信息是什么。这一过程可以表示为：

$$h_{t}=(1-z_{t})\odot h_{t-1}+z_{t} \odot \tilde h_{t-1}$$

$z_t$ 为更新门的激活结果，它同样以门控的形式控制了信息的流入。$1-z_t$ 与$ h_{t-1}$的 Hadamard 乘积表示前一时间步保留到最终记忆的信息，该信息加上当前记忆保留至最终记忆的信息($z_t$ 与$ \tilde h_{t-1}$的 Hadamard 乘积)就等于最终门控循环单元输出的内容。

以上表达式可以展示为：

![gru_final_memory](https://raw.githubusercontent.com/w5688414/paddleImage/main/gru_img/gru_final_memory.png)

门控循环单元不会随时间而清除以前的信息，它会保留相关的信息并传递到下一个单元，因此它利用全部信息而避免了梯度消失问题。

## 参考文献
Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.[链接](https://arxiv.org/pdf/1412.3555.pdf)

[经典必读：门控循环单元（GRU）的基本概念与原理](https://www.jiqizhixin.com/articles/2017-12-24)

