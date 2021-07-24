# DEEP FM算法

### 提出背景

FM(Factorization Machines)通过对于每一维特征的隐变量内积来提取特征组合，虽然理论上来讲FM可以对高阶特征组合进行建模，但实际上因为计算复杂度的原因一般都只用到了二阶特征组合。

对于高阶的特征组合，我们自然想到用DNN去解决。离散特征的处理，我们使用的是将特征转换成为one-hot的形式，但是将One-hot类型的特征输入到DNN中，会导致网络参数太多，因此使用One-hot编码作为DNN的输入是不可行的。为了解决这个问题，可以模仿FFM，将特征分为不同的field，再加上两次全连接层，让Dense Vector进行组合，那么高阶特征的组合就出来了。

<img src="images\DEEPFM1.webp" style="zoom:80%;" />

<img src="images\DEEPFM2.webp" style="zoom:80%;" />

但是低阶和高阶特征组合隐含地体现在隐藏层中，如果我们希望把低阶特征组合单独建模，然后融合高阶特征组合，即将DNN与FM进行一个合理的融合，FM负责低阶特征的提取，DNN负责高阶特征的提取。

<img src="images\DEEPFM3.webp" style="zoom:80%;" />



### DeepFM模型

DeepFM包含两部分：神经网络部分与因子分解机部分，分别负责低阶特征的提取和高阶特征的提取。这两部分**共享同样的输入**。DeepFM的预测结果可以写为：
$$
y=sigmoid(y_{FM}+y_{DNN})
$$
<img src="images\DEEPFM4.webp" style="zoom:80%;" />

#### FM部分

FM部分是一个因子分解机。因为引入了隐变量的原因，对于几乎不出现或者很少出现的隐变量，FM也可以很好的学习。
$$
y_{FM} = <w,x>+\sum_{j_1=1}^d\sum_{j_2=j_1+1}^d<v_i,v_j>x_{j1}\cdot x_{j2}
$$


FM部分的详细结构如下：

<img src="images\DEEPFM5.webp" style="zoom:80%;" />

#### DNN部分

深度部分是一个前馈神经网络。与图像或者语音这类输入不同，图像语音的输入一般是连续而且密集的，然而用于CTR的输入一般是及其稀疏的。因此需要重新设计网络结构。具体实现中为，在第一层隐含层之前，引入一个嵌入层来完成将输入向量压缩到低维稠密向量。

<img src="images\DEEPFM6.webp" style="zoom:80%;" />

嵌入层(embedding layer)的结构如下图所示。当前网络结构有两个有趣的特性，1）尽管不同field的输入长度不同，但是embedding之后向量的长度均为K。2)在FM里得到的隐变量Vik现在作为了嵌入层网络的权重。

<img src="images\DEEPFM7.webp" style="zoom:80%;" />