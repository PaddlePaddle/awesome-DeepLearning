DeepFM模型

## 模型简介

CTR预估是目前推荐系统的核心技术，其目标是预估用户点击推荐内容的概率。DeepFM模型包含FM和DNN两部分，FM模型可以抽取low-order特征，DNN可以抽取high-order特征。无需Wide&Deep模型人工特征工程。由于输入仅为原始特征，而且FM和DNN共享输入向量特征，DeepFM模型训练速度很快。

### 1）DeepFM模型

为了同时利用low-order和high-order特征，DeepFM包含FM和DNN两部分，两部分共享输入特征。对于特征i，标量wi是其1阶特征的权重，该特征和其他特征的交互影响用隐向量Vi来表示。Vi输入到FM模型获得特征的2阶表示，输入到DNN模型得到high-order高阶特征。

$$
\hat{y} = sigmoid(y_{FM} + y_{DNN})
$$

DeepFM模型结构如下图所示，完成对稀疏特征的嵌入后，由FM层和DNN层共享输入向量，经前向反馈后输出。

![](https://ai-studio-static-online.cdn.bcebos.com/8654648d844b4233b3a05e918dedc9b777cf786af2ba49af9a92fc00cd050ef3)

### 2）FM

FM模型不单可以建模1阶特征，还可以通过隐向量点积的方法高效的获得2阶特征表示，即使交叉特征在数据集中非常稀疏甚至是从来没出现过。这也是FM的优势所在。

$$
y_{FM}= <w,x> + \sum_{j_1=1}^{d}\sum_{j_2=j_1+1}^{d}<V_i,V_j>x_{j_1}\cdot x_{j_2}
$$

单独的FM层结构如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/bda8da10940b43ada3337c03332fe06ad1cd95f7780243888050023be33fc88c)

### 3）DNN

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

### 4）Loss及Auc计算

* 预测的结果将FM的一阶项部分，二阶项部分以及dnn部分相加，再通过激活函数sigmoid给出，为了得到每条样本分属于正负样本的概率，我们将预测结果和1-predict合并起来得到predict_2d，以便接下来计算auc。
* 每条样本的损失为负对数损失值，label的数据类型将转化为float输入。
* 该batch的损失avg_cost是各条样本的损失之和
* 我们同时还会计算预测的auc指标。



