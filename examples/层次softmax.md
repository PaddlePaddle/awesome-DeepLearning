## Softmax 函数
在处理多分类任务时，Softmax 回归是常用的模型，Softmax 回归( softmax regression) 又被称作多项逻辑回归( multinomial logistic regression)，它是逻辑回归在处理多类别任务上的推广。

常见的卷积神经网络 $(\mathrm{CNN})$ 或者循环神经网络 $(\mathrm{RNN})$, 如果是分类模型，最后一层通常是 Softmax 回归。
Softmax 回归的工作原理较简单，将可以判定为某类的特征相加，然后将这些特征转化为判定是这类的概率，即**把特征向量映射成概率序列**。
$$
1. p(y=i)=e^{X W_{i}^{\mathrm{T}}} / \sum_{j=0}^{k-1} e^{X W_{j}^{\mathrm{T}}+b_{j}}(1)
$$
式中: $X$ 为文本对应的特征向量；$K$ 为类别个数。
最终输出为 $K$ 维行向量, 向量的每一维都在 $(0,1)$区间中, 且总和等于 1。因此，Softmax 函数实际上是一个概率归一化函数。对应的损失函数为
$$
L(\theta)=-\log p(y=i)(2)
$$
其特点是在计算某个类别的概率时，需要对所有的类别概率做归一化处理，对于一般**小数据集、少类别的分类任务**足以满足计算速度和分类精度上的要求。当面对**数据量大、类别较多**的情况，传统的线性Softmax 就暴露出以下缺点: 

式(1)指数计算量巨大，采用随机梯度下降法 ( sto-chastic gradient descent, SGD)对式(2)进行**优化时耗时较长**，**模型的训练速度慢**。



## 基于霍夫曼树构建的层次Softmax
霍夫曼树( Huffman tree) 又称最优二叉树，是种带权路径最短的树，其特点是**权重越大，叶子节点就越靠近根节点，即权重越大，叶子节点搜索路径越短**。根据这个特性构造出的层次Softmax能够缩短目标类别的搜索路径。

假设数据集有 $M$个类别的文本，训练集中不同类别 $k_{1}, k_{2}, \cdots, k_{M}$ 的文本所对应的频数为 $\beta_{1}, \beta_{2}, \cdots, \beta_{M}$, 则利用类别频数构造的霍夫曼树流程如下:
* 【算法】 基于类别频数构造的霍夫曼树
1. 将 $\beta_{1}, \beta_{2}, \cdots, \beta_{M}$ 分别作为由 $M$ 课树组成 的森林的根节点权值。
1. 从上述根节点中选出两个权值最小的节点，作为新二叉树的左右子树,规定权值大的节点为右子树，权值小的节点为左子树，根节点的权值为左右节点权值之和。
1. 把选出的 2 颗树( 节点) 从森林中删除，并将新树加入森林。
1. 重复步骤 2、3，直到森林中只剩下一颗树为止，该树即为所求的霍夫曼树。
1. 层次Softmax 如图 1 所示，将所有文本类别的频数作为叶子节点，$M-1$ 个非叶子节点作为内部参数。其和传统的线性 $\operatorname{Softmax}$ 的区别在于计算类别标签概率的方法不同，即**将选择正确目标的过程形式化为一个迭代的二分类问题**。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/aacc838e203a46b3a6d4f1d4fe2165f9787d9e4f188a45b9857ce8b2d75b121f" width="500" hegiht="" ></center>
<center><br>图1：层次Softmax示意图</br></center>
<br></br>

设从根节点到某个叶子节点的路径长度为$L\left(k_{i}\right)$，则层次 $\operatorname{Softmax}$ 计算 $_{k i}$ 的概率的公式可表示为
$$
p\left(k_{i}\right)=\prod_{j=1}^{L\left(k_{i}\right)-1} \operatorname{sigmoid}\left\{\left[\left[n_{j+1}=L C\left(n_{j}\right)\right]\right] \cdot \varphi_{n_{j}}^{\mathrm{T}} \varepsilon\right\}(3)
$$
其中: $L C\left(n_{j}\right)$ 表示中间节点 $n_{j}$ 的左节点；$\varphi_{n_{j}}^{\mathrm{T}}$ 为 $n_{j}$ 的 参数向量的转置；$\varepsilon$ 为文本的特征向量； $\operatorname{sigmoid}(x)$ 为激活函数 $($ 逻辑回归函数)，公式为
$$
\operatorname{sigmoid}(x)=\frac{1}{1+e^{-x}}(4)
$$
指示函数 $[[X]]$ 被定义为
$$
[[X]]=\left\{\begin{array}{cl}
1 & \text { if } \quad x=\text { true } \\
-1 & \text { otherwise }
\end{array}\right.(5)
$$
于是图 1 中的 $\mathrm{k}_{2}$ 节点的概率可表示为
$$
\begin{aligned}
&p\left(k_{2}\right)=p\left(\text { left } \mid n_{1}\right) \cdot p\left(\operatorname{letf} \mid n_{2}\right) \cdot \\
&p\left(\text { right } \mid n_{3}\right)=\operatorname{sigmoid}\left(\varphi_{n_{1}}^{\mathrm{T}} \varepsilon\right) \cdot \\
&\operatorname{sigmoid}\left(\varphi_{n_{2}}^{\mathrm{T}} \varepsilon\right) \cdot \operatorname{sigmoid}\left(-\varphi_{n_{3}}^{\mathrm{T}} \varepsilon\right)
\end{aligned}(6)
$$
分析式 $(6)$ 可以发现：从根节点到叶子节点$k_{2}$，实际上进行了 3 次二分类的逻辑回归运算。

由于霍夫曼树为平衡二叉树，深度不超过 $\log _{2} M$, 因此最多计算 $\log _{2} M$ 个节点就可得到目标类别的概率值。

通过层次 $\operatorname{Softmax}$模型输出层的**计算复杂度从 $o(M)$ (计算 $M$ 次指数函数）下降到$o(\log M)$**,对相应的似然函数进行优化也会得到加速，大大**降低模型的训练时间**，但同时会**增加大量的节点参数**，在优化训练过程中需要更长迭代步，优化困难，容易过拟合( overfitting)，同时也需要更多的有效样本进行训练。
