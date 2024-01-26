二、LSTM-DSSM知识点

1、概念：针对 CNN-DSSM 无法捕获较远距离上下文特征的缺点，有人提出了用LSTM-DSSM来解决该问题。

2、模型：LSTM是一种 RNN 特殊的类型，可以学习长期依赖信息。我们分别来介绍它最重要的几个模块：

![image-20210724224028641](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224028641.png)

（0）细胞状态

细胞状态这条线可以理解成是一条信息的传送带，只有一些少量的线性交互。在上面流动可以保持信息的不变性。

![image-20210724224037901](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224037901.png)

（1）遗忘门

遗忘门 [5]由 Gers 提出，它用来控制细胞状态 cell 有哪些信息可以通过，继续往下传递。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（遗忘门）产生一个从 0 到 1 的数值 ft，然后与细胞状态 C(t-1) 相乘，最终决定有多少细胞状态可以继续往后传递。

![image-20210724224049655](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224049655.png)

（2）输入门

输入门决定要新增什么信息到细胞状态，这里包含两部分：一个 sigmoid 输入门和一个 tanh 函数。sigmoid 决定输入的信号控制，tanh 决定输入什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输入门）产生一个从 0 到 1 的数值 it，同样的信息经过 tanh 网络做非线性变换得到结果 Ct，sigmoid 的结果和 tanh 的结果相乘，最终决定有哪些信息可以输入到细胞状态里。

![image-20210724224101253](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224101253.png)

（3）输出门

输出门决定从细胞状态要输出什么信息，这里也包含两部分：一个 sigmoid 输出门和一个 tanh 函数。sigmoid 决定输出的信号控制，tanh 决定输出什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输出门）产生一个从 0 到 1 的数值 Ot，细胞状态 Ct 经过 tanh 网络做非线性变换，得到结果再与 sigmoid 的结果 Ot 相乘，最终决定有哪些信息可以输出，输出的结果 ht 会作为这个细胞的输出，也会作为传递个下一个细胞。

![image-20210724224111660](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224111660.png)

LSTM-DSSM 其实用的是 LSTM 的一个变种——加入了peephole的 LSTM。如下图所示：

![image-20210724224120533](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224120533.png)

看起来有点复杂，我们换一个图，读者可以看的更清晰：

![image-20210724224127004](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224127004.png)

这里三条黑线就是所谓的 peephole，传统的 LSTM 中遗忘门、输入门和输出门只用了 h(t-1) 和 xt 来控制门缝的大小，peephole 的意思是说不但要考虑 h(t-1) 和 xt，也要考虑 Ct-1 和 Ct，其中遗忘门和输入门考虑了 Ct-1，而输出门考虑了 Ct。总体来说需要考虑的信息更丰富了。

好了，来看一个 LSTM-DSSM 整体的网络结构：

![image-20210724224142823](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224142823.png)

红色的部分可以清晰的看到残差传递的方向。

3、作用：语义匹配

4、场景：语义分析

5、优缺点：

优点：LSTM-DSSM 通过卷积层提取了滑动窗口下的上下文信息，又通过池化层提取了全局的上下文信息，上下文信息得到较为有效的保留。

缺点：LSTM-DSSM 滑动窗口（卷积核）大小的限制，导致无法捕获该上下文信息，对于间隔较远的上下文信息，难以有效保留。