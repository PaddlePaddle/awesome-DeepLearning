**深度学习基础知识：**
**1.LSTM-DSSM概念：**
LSTM(（Long-Short-Term Memory）是一种 RNN 特殊的类型，可以学习长期依赖信息。我们分别来介绍它最重要的几个模块：
![](https://ai-studio-static-online.cdn.bcebos.com/9fcda783926b4266a0202039a804ab5322a58a740b844b37999e58c175357787)
（1）细胞状态
细胞状态这条线可以理解成是一条信息的传送带，只有一些少量的线性交互。在上面流动可以保持信息的不变性。
![](https://ai-studio-static-online.cdn.bcebos.com/63b65e50e9344611a210687b7738307c127a4db8017f4c47b37646dfcce7dc24)
（2）遗忘门
遗忘门由 Gers 提出，它用来控制细胞状态 cell 有哪些信息可以通过，继续往下传递。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（遗忘门）产生一个从 0 到 1 的数值 ft，然后与细胞状态 C(t-1) 相乘，最终决定有多少细胞状态可以继续往后传递。
![](https://ai-studio-static-online.cdn.bcebos.com/a8a363b5a0884e85aa3b1f9d35cc6c653fa00a69dca84e3c930dfb0d92871270)
（3）输入门
输入门决定要新增什么信息到细胞状态，这里包含两部分：一个 sigmoid 输入门和一个 tanh 函数。sigmoid 决定输入的信号控制，tanh 决定输入什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输入门）产生一个从 0 到 1 的数值 it，同样的信息经过 tanh 网络做非线性变换得到结果 Ct，sigmoid 的结果和 tanh 的结果相乘，最终决定有哪些信息可以输入到细胞状态里。
![](https://ai-studio-static-online.cdn.bcebos.com/aff9ced903d647fa93c8f0001106509063196218c36a485992a5c6732508c5af)
（4）输出门
输出门决定从细胞状态要输出什么信息，这里也包含两部分：一个 sigmoid 输出门和一个 tanh 函数。sigmoid 决定输出的信号控制，tanh 决定输出什么内容。如下图所示，上一层的输出 h(t-1) concat 上本层的输入 xt，经过一个 sigmoid 网络（输出门）产生一个从 0 到 1 的数值 Ot，细胞状态 Ct 经过 tanh 网络做非线性变换，得到结果再与 sigmoid 的结果 Ot 相乘，最终决定有哪些信息可以输出，输出的结果 ht 会作为这个细胞的输出，也会作为传递个下一个细胞。
![](https://ai-studio-static-online.cdn.bcebos.com/deb4d7d5dd6e4223ae1273df881ca5dea803d464228e491eac79e4dd0611faa9)

**2.LSTM-DSSM模型：**
LSTM-DSSM 其实用的是 LSTM 的一个变种——加入了peephole的 LSTM。如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/19e3220cbba045fc8d81fb5eb49d237d2734aee93ae0408c99b30079f5614e1b)
看起来有点复杂，我们换一个图可以看的更清晰：
![](https://ai-studio-static-online.cdn.bcebos.com/8b6100c76dea4c318d65078b9b9d8445f6036a5968db40d49162c937243a3dcc)

这里三条黑线就是所谓的 peephole，传统的 LSTM 中遗忘门、输入门和输出门只用了 h(t-1) 和 xt 来控制门缝的大小，peephole 的意思是说不但要考虑 h(t-1) 和 xt，也要考虑 Ct-1 和 Ct，其中遗忘门和输入门考虑了 Ct-1，而输出门考虑了 Ct。总体来说需要考虑的信息更丰富了。

下面来看一个 LSTM-DSSM 整体的网络结构：
![](https://ai-studio-static-online.cdn.bcebos.com/68e69f49a646411e9964e6eafb6b356f866df39a35f64425a7838464eeb1c8ea)
红色的部分可以清晰的看到残差传递的方向。

**3.LSTM-DSSM作用：**
针对 CNN-DSSM 无法捕获较远距离上下文特征的缺点，有人提出了用LSTM-DSSM（Long-Short-Term Memory）来解决该问题。

**4.LSTM-DSSM场景：**
以搜索引擎和搜索广告为例，最重要的也最难解决的问题是语义相似度，这里主要体现在两个方面：召回和排序。
在召回时，传统的文本相似性如 BM25，无法有效发现语义类 query-Doc 结果对，如"从北京到上海的机票"与"携程网"的相似性、"快递软件"与"菜鸟裹裹"的相似性。
在排序时，一些细微的语言变化往往带来巨大的语义变化，如"小宝宝生病怎么办"和"狗宝宝生病怎么办"、"深度学习"和"学习深度"。
DSSM（Deep Structured Semantic Models）为计算语义相似度提供了一种思路。

**5.LSTM-DSSM优缺点：**
 优点：改善了RNN中存在的长期依赖问题；LSTM的表现通常比时间递归神经网络及隐马尔科夫模型（HMM）更好；作为非线性模型，LSTM可作为复杂的非线性单元用于构造更大型深度神经网络。

[2]缺点：一个缺点是RNN的梯度问题在LSTM及其变种里面得到了一定程度的解决，但还是不够。它可以处理100个量级的序列，而对于1000个量级，或者更长的序列则依然会显得很棘手；另一个缺点是每一个LSTM的cell里面都意味着有4个全连接层(MLP)，如果LSTM的时间跨度很大，并且网络又很深，这个计算量会很大，很耗时。
