# DeepFM模型

## 1.模型简介

CTR预估是目前推荐系统的核心技术，其目标是预估用户点击推荐内容的概率。DeepFM模型包含FM和DNN两部分，FM模型可以抽取low-order（低阶）特征，DNN可以抽取high-order（高阶）特征。低阶特征可以理解为线性的特征组合，高阶特征，可以理解为经过多次线性-非线性组合操作之后形成的特征，为高度抽象特征。无需Wide&Deep模型人工特征工程。由于输入仅为原始特征，而且FM和DNN共享输入向量特征，DeepFM模型训练速度很快。

注解：Wide&Deep是一种融合浅层（wide）模型和深层（deep）模型进行联合训练的框架，综合利用浅层模型的记忆能力和深层模型的泛化能力，实现单模型对推荐系统准确性和扩展性的兼顾。

该模型的Paddle实现请参考链接：[PaddleRec版本](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/deepfm)

## 2.DeepFM模型结构

为了同时利用low-order和high-order特征，DeepFM包含FM和DNN两部分，两部分共享输入特征。对于特征i，标量wi是其1阶特征的权重，该特征和其他特征的交互影响用隐向量Vi来表示。Vi输入到FM模型获得特征的2阶表示，输入到DNN模型得到high-order高阶特征。


$$
\hat{y} = sigmoid(y_{FM} + y_{DNN})
$$

DeepFM模型结构如下图所示，完成对稀疏特征的嵌入后，由FM层和DNN层共享输入向量，经前向反馈后输出。

![](https://ai-studio-static-online.cdn.bcebos.com/8654648d844b4233b3a05e918dedc9b777cf786af2ba49af9a92fc00cd050ef3)



为什么使用FM和DNN进行结合？

* 在排序模型刚起步的年代，FM很好地解决了LR需要大规模人工特征交叉的痛点，引入任意特征的二阶特征组合，并通过向量内积求特征组合权重的方法大大提高了模型的泛化能力。
* 标准FM的缺陷也恰恰是只能做二阶特征交叉。  

所以，将FM与DNN结合可以帮助我们捕捉特征之间更复杂的非线性关系。



为什么不使用FM和RNN进行结合？

* 如果一个任务需要处理**序列**信息，即本次输入得到的输出结果，不仅和本次输入相关，还和之前的输入相关，那么使用RNN循环神经网络可以很好地利用到这样的**序列**信息
* 在预估点击率时，我们会假设用户每次是否点击的事件是独立的，不需要考虑**序列**信息，因此RNN于FM结合来预估点击率并不合适。还是使用DNN来模拟出特征之间的更复杂的非线性关系更能帮助到FM。

## 3.FM

FM（Factorization Machines，因子分解机）最早由Steffen Rendle于2010年在ICDM上提出，它是一种通用的预测方法，在即使数据非常稀疏的情况下，依然能估计出可靠的参数进行预测。与传统的简单线性模型不同的是，因子分解机考虑了特征间的交叉，对所有嵌套变量交互进行建模（类似于SVM中的核函数），因此在推荐系统和计算广告领域关注的点击率CTR（click-through rate）和转化率CVR（conversion rate）两项指标上有着良好的表现。

为什么使用FM？

* 特征组合是许多机器学习建模过程中遇到的问题，如果对特征直接建模，很有可能忽略掉特征与特征之间的关联信息，一次可以通过构建新的交叉特征这一特征组合方式提高模型的效果。FM可以得到特征之间的关联信息。
* 高维的稀疏矩阵是实际工程中常见的问题，并且直接导致计算量过大，特征权值更新缓慢。试想一个10000100的表，每一列都有8中元素，经过one-hot编码之后，会产生一个10000800的表。  

而FM的优势就在于对这两方面问题的处理。首先是特征组合，通过两两特征组合，引入交叉项特征（二阶特征），提高模型得分；其次是高维灾难，通过引入隐向量（对参数矩阵进行分解），完成特征参数的估计。

FM模型不单可以建模1阶特征，还可以通过隐向量点积的方法高效的获得2阶特征表示，即使交叉特征在数据集中非常稀疏甚至是从来没出现过。这也是FM的优势所在。


$$
y_{FM}= <w,x> + \sum_{j_1=1}^{d}\sum_{j_2=j_1+1}^{d}<V_i,V_j>x_{j_1}\cdot x_{j_2}
$$

单独的FM层结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/bda8da10940b43ada3337c03332fe06ad1cd95f7780243888050023be33fc88c)



## 4.DNN

该部分和Wide&Deep模型类似，是简单的前馈网络。在输入特征部分，由于原始特征向量多是高纬度,高度稀疏，连续和类别混合的分域特征，因此将原始的稀疏表示特征映射为稠密的特征向量。

假设子网络的输出层为：


$$
a^{(0)}=[e1,e2,e3,...en]
$$
DNN网络第l层表示为：


$$
a^{(l+1)}=\sigma{（W^{(l)}a^{(l)}+b^{(l)}）}
$$
再假设有H个隐藏层，DNN部分的预测输出可表示为：


$$
y_{DNN}= \sigma{(W^{|H|+1}\cdot a^H + b^{|H|+1})}
$$
DNN深度神经网络层结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/df8159e1d56646fe868e8a3ed71c6a46f03c716ad1d74f3fae88800231e2f6d8)



## 5.Loss及Auc计算

DeepFM模型的损失函数选择Binary_Cross_Entropy（二值交叉熵）函数


$$
H_p(q)=-\frac{1}{N}\sum_{i=1}^Ny_i\cdot log(p(y_i))+(1-y_i) \cdot log(1-p(y_i))
$$
对于公式的理解，y是样本点，p(y)是该样本为正样本的概率，log(p(y))可理解为对数概率。

Auc是Area Under Curve的首字母缩写，这里的Curve指的就是ROC曲线，AUC就是ROC曲线下面的面积,作为模型评价指标，他可以用来评价二分类模型。其中，ROC曲线全称为受试者工作特征曲线 （receiver operating characteristic curve），它是根据一系列不同的二分类方式（分界值或决定阈），以真阳性率（敏感性）为纵坐标，假阳性率（1-特异性）为横坐标绘制的曲线。

可使用paddle.metric.Auc()进行调用。

可参考已有的资料：[机器学习常用评估指标](https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/metrics/evaluation_metric.html?highlight=auc#auc)



## 6.与其他模型的对比



![](https://ai-studio-static-online.cdn.bcebos.com/09f6e16a0ca74b82ba19c92d765244927c89aa48b5fa4574a0db292a9567b176)

如表1所示，关于是否需要预训练，高阶特征，低阶特征和是否需要特征工程的比较上，列出了DeepFM和其他几种模型的对比。DeepFM表现更优。

![](https://ai-studio-static-online.cdn.bcebos.com/6259b8c917484ae0893ece1b2ac45ffeae5e567728984c58a3a99a78314047d5)



如表2所示，不同模型在Company*数据集和Criteo数据集上对点击率CTR进行预估的性能表现。DeepFM在各个指标上表现均强于其他模型。

## 7.参考文献

[[IJCAI 2017]Guo, Huifeng，Tang, Ruiming，Ye, Yunming，Li, Zhenguo，He, Xiuqiang. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)



