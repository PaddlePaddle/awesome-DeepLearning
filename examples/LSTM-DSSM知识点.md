# 二、LSTM-DSSM知识点补充
针对 CNN-DSSM 无法捕获较远距离上下文特征的缺点，有人提出了用LSTM-DSSM[3]（Long-Short-Term Memory）来解决该问题。不过说 LSTM 之前，要先介绍它的"爸爸""RNN。

**4.1 RNN**
RNN（Recurrent Neural Networks）可以被看做是同一神经网络的多次复制，每个神经网络模块会把消息传递给下一个。如果我们将这个循环展开：
![](https://ai-studio-static-online.cdn.bcebos.com/0de458bb34204d209839707a446294949f0b2f552fe54cbb8a5800cb97a5a20d)
假设输入 xi 为一个 query 中几个连续的词，hi 为输出。那么上一个神经元的输出 h(t-1) 与当前细胞的输入 Xt 拼接后经过 tanh 函数会输出 ht，同时把 ht 传递给下一个细胞。
![](https://ai-studio-static-online.cdn.bcebos.com/0de8cda449214bfeb3d4216649896ccf62a58b2d94a14ef29dfc12ca898560ee)
不幸的是，在这个间隔不断增大时，RNN 会逐渐丧失学习到远距离信息的能力。因为 RNN 随着距离的加长，会导致梯度消失。简单来说，由于求导的链式法则，直接导致梯度被表示为连乘的形式，以至梯度消失（几个小于 1 的数相乘会逐渐趋向于 0）。

**4.2 LSTM**
LSTM[4](（Long-Short-Term Memory）是一种 RNN 特殊的类型，可以学习长期依赖信息。我们分别来介绍它最重要的几个模块：
![](https://ai-studio-static-online.cdn.bcebos.com/45ab54ee9be94774b48de791efd341c08c116d124f184fa5870254b94f722ecc)
（0）细胞状态

细胞状态这条线可以理解成是一条信息的传送带，只有一些少量的线性交互。在上面流动可以保持信息的不变性。

![](https://ai-studio-static-online.cdn.bcebos.com/1c6a5d82d39f43f6a24eabeb33a00acde517c9c42e0043e2882416a3e103aee8)
（1）遗忘门

遗忘门 [5]由 Gers 提出，它用来控制细胞状态 cell 有哪些信息可以通过，继续往下传递。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（遗忘门）产生一个从 0 到 1 的数值 ft，然后与细胞状态 C(t-1) 相乘，最终决定有多少细胞状态可以继续往后传递。
![](https://ai-studio-static-online.cdn.bcebos.com/3a92fb1489424771919562c7dec314633eca02b09ec8444dabd66eaccf8ad703)
（2）输入门

输入门决定要新增什么信息到细胞状态，这里包含两部分：一个 sigmoid 输入门和一个 tanh 函数。sigmoid 决定输入的信号控制，tanh 决定输入什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输入门）产生一个从 0 到 1 的数值 it，同样的信息经过 tanh 网络做非线性变换得到结果 Ct，sigmoid 的结果和 tanh 的结果相乘，最终决定有哪些信息可以输入到细胞状态里。
![](https://ai-studio-static-online.cdn.bcebos.com/0c63c31a07374282a4c38341db0512634fba5ccef3ff41a4984883c52310734c)
（3）输出门

输出门决定从细胞状态要输出什么信息，这里也包含两部分：一个 sigmoid 输出门和一个 tanh 函数。sigmoid 决定输出的信号控制，tanh 决定输出什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输出门）产生一个从 0 到 1 的数值 Ot，细胞状态 Ct 经过 tanh 网络做非线性变换，得到结果再与 sigmoid 的结果 Ot 相乘，最终决定有哪些信息可以输出，输出的结果 ht 会作为这个细胞的输出，也会作为传递个下一个细胞。

![](https://ai-studio-static-online.cdn.bcebos.com/3c227f39f4964e2b9c306a1ec14456a39414d98f53cb4a41ac50315c7196a65f)
**4.3 LSTM-DSSM**
LSTM-DSSM 其实用的是 LSTM 的一个变种——加入了peephole[6]的 LSTM。如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/255b43568c264303981be913d9e5045898ebf3925eee4e698b21ba0c0d04f47b)
看起来有点复杂，我们换一个图，可以看的更清晰：
![](https://ai-studio-static-online.cdn.bcebos.com/3078adb7908c4aa8bfbba01b6d8f82ba350d31e3b6bf455bbe145eb839e1053f)
这里三条黑线就是所谓的 peephole，传统的 LSTM 中遗忘门、输入门和输出门只用了 h(t-1) 和 xt 来控制门缝的大小，peephole 的意思是说不但要考虑 h(t-1) 和 xt，也要考虑 Ct-1 和 Ct，其中遗忘门和输入门考虑了 Ct-1，而输出门考虑了 Ct。总体来说需要考虑的信息更丰富了。

好了，来看一个 LSTM-DSSM 整体的网络结构：
![](https://ai-studio-static-online.cdn.bcebos.com/a8562aa0f9714a8cab409032984ae100390ba898051e418db4dd41ccb1e710c2)

