## CTR预估
CTR预估是对每次广告的点击情况做出预测，预测用户是点击还是不点击。


CTR预估数据特点：

* 输入中包含类别型和连续型数据。类别型数据需要one-hot,连续型数据可以先离散化再one-hot，也可以直接保留原值
* 维度非常高
* 数据非常稀疏
* 特征按照Field分组

CTR预估重点在于学习组合特征。组合特征包括二阶、三阶甚至更高阶的，阶数越高越复杂，越不容易学习。Google的论文研究得出结论：高阶和低阶的组合特征都非常重要，同时学习到这两种组合特征的性能要比只考虑其中一种的性能要好。

那么关键问题转化成：如何高效的提取这些组合特征。一种办法就是引入领域知识人工进行特征工程。这样做的弊端是高阶组合特征非常难提取，会耗费极大的人力。而且，有些组合特征是隐藏在数据中的，即使是专家也不一定能提取出来，比如著名的“尿布与啤酒”问题。

在DeepFM提出之前，已有LR，FM，FFM，FNN，PNN（以及三种变体：IPNN,OPNN,PNN*）,Wide&Deep模型，这些模型在CTR或者是推荐系统中被广泛使用。

## DeepFM - CTR
近年来深度学习模型在解决NLP、CV等领域的问题上取得了不错的效果，于是有学者将深度神经网络模型与FM模型结合，提出了DeepFM模型。

FM通过对于每一位特征的隐变量内积来提取特征组合，有着以下优点：
1. FM模型的参数支持非常稀疏的特征，而SVM等模型不行
1. FM的时间复杂度为 $o(N)$，并且可以直接优化原问题的参数，而不需要依靠支持向量或者是转化成对偶问题解决
1. FM是通用的模型，可以适用于任何实数特征的场景，其他的模型不行

但FM通常使用二维特征的交叉，因为当特征数量>2时，没有很好的优化方法，同时三重特征的交叉往往没有意义，且会过于稀疏，由此可见FM适用于低维特征交叉。对于高维特征组合来说，我们很自然想到深度神经网络DNN。而将这二者结合，便是DeepFM模型。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/571f76ff3c97446385da7fa6c9e8fc2599399d9e910c41ce93121d0382238112" width="500" hegiht="" ></center>
<center><br>图1：DeepFM模型</br></center>
<br></br>

DeepFM的架构如下:
* 输入的是稀疏特征的id
* 进行一层lookup 之后得到id的稠密embedding
* 这个embedding一方面作为隐向量输入到FM层进行计算
* 同时该embedding进行聚合之后输入到一个DNN模型(deep)
* 然后将FM层和DNN层的输入求和之后进行co-train

DeepFM目的是同时学习低阶和高阶的特征交叉，主要由FM和DNN两部分组成，底部共享同样的输入。模型可以表示为：
$$
\hat{y}=\operatorname{sigmoid}\left(y_{F M}+y_{D N N}\right)
$$

### **FM部分**
在线性回归的基础上，将二维特征进行组合，并通过向量的内积得到了交叉特征的系数，数学表达为：
$$
y_{F M}=<w, x>+\sum_{i=1}^{d} \sum_{j=i+1}^{d}<V_{i}, V_{j}>x_{i} \cdot x_{j}
$$
通过对该公式的推导化简，FM可以将计算复杂度降至$o(kN)$：
$$
\begin{aligned}
\sum_{i=1}^{n} \sum_{j=i+1}^{n} v_{i}^{T} v_{j} x_{i} x_{j} &=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} v_{i}^{T} v_{j} x_{i} x_{j}-\frac{1}{2} \sum_{i=1}^{n} v_{i}^{T} v_{j} x_{i} x_{j} \\
&=\frac{1}{2}\left(\sum_{i=1}^{n} \sum_{j=1}^{n} \sum_{f=1}^{k} v_{i, f} v_{j, f} x_{i} x_{j}-\sum_{i=1}^{n} \sum_{f=1}^{k} v_{i, f} v_{i, f} x_{i} x_{i}\right) \\
&=\frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)\left(\sum_{j=1}^{n} v_{j, f} x_{j}\right)-\sum_{i=1}^{n} v_{i, f}^{2} x_{i}^{2}\right) \\
&=\frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i, f} x_{i}\right)^{2}-\sum_{i=1}^{n} v_{i, f}^{2} x_{i}^{2}\right)
\end{aligned}
$$
推导过程如下，第一、二行是把 $v_{i}^{T} v_{j}$ 向量内积展开。第二行到第三行有三个 $\Sigma$，可以提取出的是最里面的 $\Sigma$，因为是有限项求和，所以调换顺序并不会影响结果。提取出公因式后得到的结果是两个平方项。观察一下即可发现，这两个平方项的计算复杂度都是 $O(N)$, 再加上外面一层 $O(k)$ 的复杂度，整体的复杂度是 $O(kN)$。这便是FM的优化过程。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/d5d39994892644d0b8aaac6ce61cd46cb6aeee7ad55d42688a44e19cd83e1eeb" width="500" hegiht="" ></center>
<center><br>图2：FM部分</br></center>
<br></br>

### **Deep部分**
深度部分是一个前馈神经网络，与图像或语音类的输入不同，CTR的输入一般是极其稀疏的，因此需要重新设计网络结构。在第一层隐藏层之前，引入一个嵌入层来完成输入向量压缩到低位稠密向量：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/45125474c1924dbc9baaa61a323caa04520c6c12d9f949849af410cdd5b07ba4" width="500" hegiht="" ></center>
<center><br>图3：Deep部分嵌入层结构</br></center>
<br></br>

嵌入层的结构如图3所示，有两个特性：
- 尽管不同field的输入长度不同，但是embedding之后向量的长度均为k
- 在FM中得到的隐变量 $V_{i k}$ 现在作为嵌入层网络的权重
嵌入层的输出为 $a^{(0)}=\left[e_{1}, e_{2}, \ldots, e_{m}\right]$, 其中 $e_{i}$ 是嵌入的第i个filed, $\mathrm{m}$ 是 field的个数，前向过程将嵌入层的输出输入到隐藏层为：
$$
a^{(l+1)}=\sigma\left(W^{(l)} a^{(l)}+b^{(l)}\right)
$$
其中l是层数， $\sigma$ 是激活函数， $W^{(l)}$ 是模型的权重， $b^{(l)}$ 是1层的偏置， 因此, $\mathrm{DNN}$ 得预测模型表达为：
$$
y_{D N N}=\sigma\left(W^{|H|+1} \cdot a^{H}+b^{|H|+1}\right)
$$
其中， $|H|$ 为隐藏层层数。

### **DeepFM模型对比**
目前在推荐领域中比较流行的深度模型有FNN、PNN、Wide&Deep。

* FNN模型是用FM模型来对Embedding层进行初始化的全连接神经网络。
* PNN模型则是在Embedding层和全连接层之间引入了内积/外积层，来学习特征之间的交互关系。
* Wide&Deep模型由谷歌提出，将LR和DNN联合训练，在Google Play取得了线上效果的提升。Wide&Deep模型，很大程度上满足了模型同时学习低阶特征和高阶特征的需求，让模型同时具备较好的“memorization”和“generalization”。但是需要人工特征工程来为Wide模型选取输入特征。具体而言，对哪些稀疏的特征进行embedding，是由人工指定的。

有学者将DeepFM与当前流行的应用于CTR的神经网络做了对比：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f78ba362c0fa44e3a151dcc75b69b6a0ef181a9788a0467a91f50437deba48ba" width="900" hegiht="" ></center>
<center><br>图4：DeepFM与其他模型对比</br></center>
<br></br>

从预训练，特征维度以及特征工程的角度进行对比，发现：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/7422c7556043419eb4ba2dc76a452067306007d6a28149199e608861473fe76f" width="500" hegiht="" ></center>
<center><br>图5：预训练、特征维度及特征工程的角度对比</br></center>
<br></br>

从实验效果来看，DeepFM的效果较好：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/557a21d7691e40069631385d8cbf6441da8a0d7fe93448bc973730788a0fec50" width="500" hegiht="" ></center>
<center><br>图6：实验结果对比</br></center>
<br></br>


DeepFM的三大优势：

1. 相对于Wide&Deep不再需要手工构建wide部分；
1. 相对于FNN把FM的隐向量参数直接作为网络参数学习；
1. DeepFM将embedding层结果输入给FM和MLP，两者输出叠加，达到捕捉了低阶和高阶特征交叉的目的。

参考文献：Guo, Huifeng, et al. “DeepFM: a factorization-machine based neural network for CTR prediction.” arXiv preprint arXiv:1703.04247 (2017).
