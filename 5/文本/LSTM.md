**LSTM原理：**
1.LSTM简介：
长短期记忆（Long short-term memory, LSTM）是一种特殊的RNN，主要是为了解决长序列训练过程中的梯度消失和梯度爆炸问题。简单来说，就是相比普通的RNN，LSTM能够在更长的序列中有更好的表现。
LSTM结构和普通RNN的主要输入输出区别如下所示：
![](https://ai-studio-static-online.cdn.bcebos.com/688399fe22f1423e9d7b1c5f22ce9f93dc87680302d5428584e86ecbbb0393c0)
相比RNN只有一个传递状态ht，LSTM有两个传输状态，一个ct（cell state），和一个 ht(hidden state）。（Tips：RNN中的ht对于LSTM中的ct）
其中对于传递下去的ct改变得很慢，通常输出的ct是上一个状态传过来的c(t-1)加上一些数值。
而ht则在不同节点下往往会有很大的区别。
2.LSTM原理：
下面具体对LSTM的内部结构来进行剖析。
首先使用LSTM的当前输入xt和上一个状态传递下来的h(t-1)拼接训练得到四个状态。
![](https://ai-studio-static-online.cdn.bcebos.com/6d181454198d415bb6102067ae87533de1c48212b7594ba78ffe483bbd0feb8f)
![](https://ai-studio-static-online.cdn.bcebos.com/3dde30f469974cfba0acdf542db9d1996d237dd3d87448b49075453c521528f9)
其中，zf,zi,zo是由拼接向量乘以权重矩阵之后，再通过一个sigmoid激活函数转换成0到1之间的数值，来作为一种门控状态。
而z则是将结果通过一个tanh激活函数将转换成-1到1之间的值（这里使用tanh是因为这里是将其做为输入数据，而不是门控信号）。

下面开始进一步介绍这四个状态在LSTM内部的使用。
![](https://ai-studio-static-online.cdn.bcebos.com/6b5e3221d3e6448b9c5a9e1b7bb75d0104ae0ce1c889448cb5422cc1fba3faa2)
途中两个符号，一个代表进行矩阵加法；另一个是Hadamard Product，也就是操作矩阵中对应的元素相乘，因此要求两个相乘矩阵是同型的。

LSTM内部主要有三个阶段：
1. 忘记阶段。这个阶段主要是对上一个节点传进来的输入进行选择性忘记。简单来说就是会 “忘记不重要的，记住重要的”。
具体来说是通过计算得到的zf（f表示forget）来作为忘记门控，来控制上一个状态的 c(t-1)哪些需要留哪些需要忘。
2. 选择记忆阶段。这个阶段将这个阶段的输入有选择性地进行“记忆”。主要是会对输入xt进行选择记忆。哪些重要则着重记录下来，哪些不重要，则少记一些。当前的输入内容由前面计算得到的z表示。而选择的门控信号则是由zi（i代表information）来进行控制。
将上面两步得到的结果相加，即可得到传输给下一个状态的ct。也就是上图中的第一个公式。
3. 输出阶段。这个阶段将决定哪些将会被当成当前状态的输出。主要是通过zo来进行控制的。并且还对上一阶段得到的co进行了放缩（通过一个tanh激活函数进行变化）。
与普通RNN类似，输出yt往往最终也是通过ht变化得到。
