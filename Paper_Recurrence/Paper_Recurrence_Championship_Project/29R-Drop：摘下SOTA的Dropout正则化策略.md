# R-Drop：摘下SOTA的Dropout正则化策略

一种比Dropout更简单又有效的正则化方法

# 摘要

Dropout是一种功能强大且广泛应用的技术，用于正则化深度神经网络的训练。基于此，微软亚洲研究院与苏州大学提出了更加简单有效的正则方法 R-Drop（Regularized Dropout）。

与传统作用于神经元（Dropout）或者模型参数（DropConnect）上的约束方法不同，R-Drop作用于模型的输出层，弥补了Dropout在训练和测试时的不一致性。简单来说就是在每个 mini-batch 中，每个数据样本两次经过带有Dropout的同一个模型，再使用KL-divergence约束，使两次的输出一致。所以，R-Drop可以使Dropout带来的两个随机子模型有一致的输出。

与传统的训练方法相比，R-Drop 只是简单增加了一个KL-divergence损失函数，并没有其他任何改动。虽然该方法看起来很简单，但实验表明，R-Drop 在5个常用的包含 NLP（自然语言处理） 和 CV（计算机视觉） 的任务中都取得了当前最优的结果。

# 一、介绍

深度神经网络（DNN）近来已经在各个领域都取得了令人瞩目的成功。在训练这些大规模的 DNN 模型时，正则化技术，如 L2 Normalization、Batch Normalization、Dropout 等是不可缺少的模块，以防止模型过拟合，同时提升模型的泛化能力。在这其中，Dropout技术由于只需要简单地在训练过程中丢弃一部分的神经元，而成为了被最广为使用的正则化技术。

与传统作用于神经元（Dropout）或者模型参数（DropConnect）上的约束方法不同，R-Drop作用于模型的输出层，弥补了Dropout在训练和测试时的不一致性。简单来说就是在每个 mini-batch 中，每个数据样本两次经过带有Dropout的同一个模型，再使用KL-divergence约束，使两次的输出一致。所以，R-Drop可以使Dropout带来的两个随机子模型有一致的输出。

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/a2f149f656c74622a2244b857af1cfd3ad1f049969bb440d8cb0325bbcf1f7fe" width = "80%" height = "80%" />
<center>图1：R-Drop总体框架。作者以Transformer结构为例。左图显示，一个输入x将通过模型两次，并获得两个分布P1和P2，而右图显示了由Dropout产生的两个不同的子模型。</center>

作者从优化器的角度对R-Drop的正则化效应进行了理论分析，表明R-Drop隐式地正则化了参数空间的自由度，从而降低了模型空间的复杂度，增强了模型的泛化能力。

这篇文章的的主要贡献总结如下：
- 作者提出了R-Drop，这是一种基于Drop的简单而有效的正则化方法，可普遍应用于训练不同类型的深层模型。
- 作者从理论上证明，R-Drop可以减少模型参数的自由度，这与处理隐藏单元或模型权重的其他正则化方法是互补的。
- 通过对总共18个数据集的4个NLP和1个CV任务的广泛实验，作者表明R-Drop实现了极其强大的性能，包括多个SOTA结果。

# 二、方法

R-Drop正则化方法的总体框架如图1所示。在详细说明细节之前，作者首先给出一些必要的符号。给定训练数据集$D={\left\{ (x_i，y_i)\right\}}^n_{i=1}$，训练的目标是学习模型$P^w（y|x）$，其中$n$是训练样本的数量，$(x_i，y_i)$是标记的数据对。$x_i$是输入数据，$y_i$是标签。例如，在NLP中，$x_i$可以是机器翻译中的源语言句子，而$y_i$是对应的目标语言句子。在CV中，$x_i$可以是一张图像，$y_i$是该图像的类标签。映射函数的概率分布也表示为$P^w（y|x）$，两个分布$P_1$和$P_2$之间的Kullback-Leibler（KL）散度用$D_{KL}（P_1||P_2）$表示。下面，作者分别解释了R-Drop、训练算法和理论分析。

## 1.R-Drop正则化

给定训练数据$D={\left\{ (x_i，y_i)\right\}}^n_{i=1}$，深度学习模型的一个基本学习目标是最小化负对数似然损失函数，如下所示：

$$
\mathcal{L}_{n l l}=\frac{1}{n} \sum_{i=1}^{n}-\log \mathcal{P}^{w}\left(y_{i} \mid x_{i}\right) （1）
$$

由于深度神经网络容易出现过拟合现象，在训练过程中通常采用dropout等正则化方法来减小模型的泛化误差。具体地说，dropout在神经网络的每一层中随机丢弃部分单元，以避免共同适应和过度拟合。此外，dropout还可以近似地将许多不同的神经网络结构以指数形式有效地组合起来，而模型组合总是可以提高模型性能。基于上述特征和 dropout带来的结构随机性，作者提出了R-Drop来进一步规范 dropout子模型的输出预测。

具体地，在每个训练步骤中给出输入数据$x_i$，我们给$x_i$进行2次前向传播。因此，可以得到模型预测的两个分布，表示为$P^w_1（y_i|x_i）$和$P^w_2（y_i|x_i）$。如上所述，由于dropout操作在一个模型中随机丢弃单元，因此两个前向传播确实基于两个不同的子模型（尽管在同一个模型中）。如图1的右部分所示，输出预测$P^w_1（y_i|x_i）$的左路径的每一层中的dropout单元与输出分布$P^w_2（y_i|x_i）$的右路径的丢弃单元不同。因此，对于相同的输入数据对$（x_i，y_i）$，$P^w_1（y_i|x_i）$和$P^w_2（y_i|x_i）$的分布是不同的。然后，在该训练步骤中，作者的R-Drop方法通过最小化同一样本的这两个输出分布之间的双向Kullback-Leibler（KL）散度，尝试对模型预测进行正则化，即：

$$
\mathcal{L}_{K L}^{i}=\frac{1}{2}\left(\mathcal{D}_{K L}\left(\mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right) \| \mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right)\right)+\mathcal{D}_{K L}\left(\mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right) \| \mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right)\right)\right) （2）
$$

使用两个前向传播的基本负对数似然学习目标$L^i_{NLL}$：

$$
\mathcal{L}_{N L L}^{i}=-\log \mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right)-\log \mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right) （3）
$$

最终的训练目标是尽量减少数据$（x_i，y_i）$的$L^i$：

$$
\begin{aligned}
\mathcal{L}^{i}=\mathcal{L}_{N L L}^{i}+\alpha \cdot \mathcal{L}_{K L}^{i}=&-\log \mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right)-\log \mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right) +\frac{\alpha}{2}\left[\mathcal{D}_{K L}\left(\mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right) \| \mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right)\right)+\mathcal{D}_{K L}\left(\mathcal{P}_{2}^{w}\left(y_{i} \mid x_{i}\right) \| \mathcal{P}_{1}^{w}\left(y_{i} \mid x_{i}\right)\right)\right]
\end{aligned} （4）
$$

其中$α$是控制$L^i_{KL}$的系数权重。通过这种方式，R-Drop进一步将模型空间正则化，超越了dropout，并提高了模型的泛化能力。与方程（1）和方程（4）相比，R-Drop只增加了基于训练中两次前向传播的KL发散损失$L^i_{KL}$。请注意，如果模型中存在可产生不同子模型或输出的随机性（例如，dropout），则作者的正则化方法可普遍应用于不同的模型结构。

## 2.训练算法

基于R-Drop的整体训练算法在算法1中给出。如前所述，在每个训练步骤中，第3-5行显示作者将模型前向传播两次，获得输出分布$P^w_1（y_i|x_i）$和$P^w_2（y_i|x_i）$，然后第6-7行计算负对数似然和两个分布之间的KL散度。最后，根据方程式（4）的损失更新模型参数（第8行）。训练将在整个数据集持续进行，直至收敛。为了节省训练成本，我们不进行两次前向传播，而是重复输入一个$x$，并将它们连接起来$（[x；x]）$在同一小批次中传播一次。与传统的训练相比，作者的实现类似于将批量大小扩大一倍，一个潜在的限制是R-Drop的计算成本在每一步都会增加。正如作者在第4.1节中所示，与其他正则化方法（例如，训练w/或w/o dropout）类似，尽管R-Drop需要更多的训练才能收敛，但最终的优化效果要好得多，性能也更好。作者还在附录C.1中展示了另一项基线研究，该研究的batch size增加了一倍。

<img style="display: block; margin: 0 auto;" src="https://ai-studio-static-online.cdn.bcebos.com/89764b21850f4dbfb1c1070ec92327b03d44369576324a58824408dfa710830b" width = "80%" height = "80%" />

## 3.理论分析

在本小节中，作者分析了R-Drop的正则化效应。设$h^l（x）∈ R^d$表示具有输入向量$x$的神经网络第l层的输出，并设$ξ^l∈ R^d$表示一个随机向量，其每个维度从伯努利分布$B（p）$中独立采样：

$$
\xi_{i}^{l}= \begin{cases}1, & \text { with probability } p, \\ 0, & \text { with probability } 1-p .\end{cases}
$$

那么$h^l（x）$上的dropout操作可以用$h_{\xi^{l}}^{l}(x)=\frac{1}{p} \xi^{l} \odot h^{l}(x)$表示，其中$\odot$表示元素的乘积。因此，应用dropout后，参数为$w$的神经网络的输出分布为$\mathcal{P}_{\xi}^{w}(y \mid x):=\operatorname{softmax}\left(\text { linear }\left(h_{\xi^{L}}^{L}\left(\cdots\left(h_{\xi^{1}}^{1}\left(x_{\xi^{0}}\right)\right)\right)\right)\right)$，其中$\xi=\left(\xi^{L}, \cdots, \xi^{0}\right)$。R-Drop增强训练可表示为解决以下约束优化问题：

$$
\begin{aligned}
&\min _{w} \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}_{\xi}\left[-\log \mathcal{P}_{\xi}^{w}\left(y_{i} \mid x_{i}\right)\right] （5）\\
&\text { s.t. } \left.\quad \frac{1}{n} \sum_{i=1}^{n} \mathbb{E}_{\xi^{(1)}, \xi^{(2)}}\left[\mathcal{D}_{K L}\left(\mathcal{P}_{\xi^{(1)}}^{w}\left(y_{i} \mid x_{i}\right) \| \mathcal{P}_{\xi^{(2)}}^{w}\left(y_{i} \mid x_{i}\right)\right)\right)\right]=0 .（6）
\end{aligned}
$$

更准确地说，R-Drop以随机方式约束优化问题，即，它从伯努利分布和一个训练实例$（x_i，y_i）$中采样两个随机向量$ξ_{(1)}$和$ξ_{(1)}$（对应于两个dropout实例），并根据随机梯度更新参数。

与不使用dropout的损失$\mathcal{L}=\frac{1}{n} \sum_{i=1}^{n}-\log \mathcal{P}^{w}\left(y_{i} \mid x_{i}\right)$相比，优化损失$L_{NLL}$通过控制模型$\mathcal{P}_{\xi}^{w}(\cdot)$的雅可比矩阵来限制模型的复杂性。

实际上，约束神经网络任意两个子结构的KL散度会对神经网络参数的自由度产生约束。因此，约束优化的问题转而寻求一个模型，该模型可以在参数自由度最小的情况下最小化损失$L_{NLL}$，从而避免过拟合并提高泛化能力。

# 三、实验

实验部分我使用脚本任务进行多卡训练，论文中训练了10000个step，训练结束时模型已经收敛，且达到了论文精度：

![](https://ai-studio-static-online.cdn.bcebos.com/8e592da2aa1b4cfeb33ee436d1b6cbbef42440585133490990203be87b5ee4dc)

log可视化我已上传至服务器，可随时查看：[http://180.76.144.223:8040/app/scalar](http://180.76.144.223:8040/app/scalar)

我把模型权重放在了[https://aistudio.baidu.com/aistudio/datasetdetail/105204](https://aistudio.baidu.com/aistudio/datasetdetail/105204)，并加载到该项目中，以便检验精度：

## 安装依赖库


```python
!pip install ml_collections
```

    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Collecting ml_collections
    [?25l  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/03/d4/9ab1a8c2aebf78c348404c464733974dc4e7088174d6272ed09c2fa5a8fa/ml_collections-0.1.0-py3-none-any.whl (88kB)
    [K     |████████████████████████████████| 92kB 401kB/s eta 0:00:011
    [?25hRequirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ml_collections) (1.15.0)
    Collecting contextlib2 (from ml_collections)
      Downloading https://pypi.tuna.tsinghua.edu.cn/packages/76/56/6d6872f79d14c0cb02f1646cbb4592eef935857c0951a105874b7b62a0c3/contextlib2-21.6.0-py2.py3-none-any.whl
    Requirement already satisfied: PyYAML in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ml_collections) (5.1.2)
    Requirement already satisfied: absl-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from ml_collections) (0.8.1)
    Installing collected packages: contextlib2, ml-collections
    Successfully installed contextlib2-21.6.0 ml-collections-0.1.0


## 精度验证

验证模型在CIFAR100上的精度：


```python
!python work/test.py --name cifar100-test
```

    W1013 07:52:09.326622 29326 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W1013 07:52:09.330857 29326 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    Validating... (loss=X.X):   0%|| 0/157 [00:00<?, ?it/s]/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    Validating... (loss=0.28328): 100%|| 157/157 [02:20<00:00,  1.12it/s]
    accuracy: 0.93480


论文《R-Drop: Regularized Dropout for Neural Networks》要求的数据集是CIFAR-100，验收标准是ViT-B_16+RD在CIFAR100的验证集上准确率为93.29%，我的复现精度为93.48%比论文的精度高0.1个点左右。（脚本任务训练完以后的模型最高精度是93.84%，但是把模型拿下来放在单卡跑的时候，精度有所损失）

完整代码我已上传至GitHub，链接为：[https://github.com/zbp-xxxp/R-Drop-Paddle](https://github.com/zbp-xxxp/R-Drop-Paddle)

## 参数的设置

1. **关于batchsize**：原论文作者使用512的batchsize做训练。在pytorch的多卡训练中，这是所有卡总的batchsize；而在paddle中，设置的是单卡的batchsize，因此使用脚本任务四卡训练时，应该把batchsize设为128，这样总的batchsize才是128×4=512。

2. **关于多卡并行训练**：多卡分布式训练时，数据处理部分需要加上`distributebatchsampler`，这样相当于把数据分到多个卡上训练，否则其实就是每个卡都训练一遍全量数据

3. **关于迭代次数**：官方给的迭代次数是10000，即1后面4个0。我最开始的时候看错了，多了一个0，因为学习率是根据迭代次数算的，所以迭代次数错了，学习率也会跟着错

4. **关于学习率**：一开始，我们为了让batchsize对齐原论文，一直在调学习率，但是，当我们解决上面三个问题后，其实就不需要对学习率做调整了

5. **关于输入尺寸**：对于输入尺寸，我现在还是有疑问。cifar100的图像大小是32，如果要resize到384，就需要插值，又或者是在图像四周填充0，这样的做法：
    - 会丢失像素之间的关联性
    - 模型会学到因为插值而带来的噪声

    但是我们后来按照384的输入去训练时，确实是能让模型达到一个比较好的效果，但至于这样做的可解释性，我目前还是很疑惑。

# 四、总结

R-Drop这篇论文解决了Dropout在训练与预测时输出不一致的问题，论文作者将解决该问题的方法取名为R-drop，这是一种基于dropout的简单而有效的正则化方法，它通过在模型训练中最小化从dropout中采样的任何一对子模型的输出分布的双向KL散度来实现。最核心的代码如下所示：

```
import paddle
import paddle.nn.functional as F

class kl_loss(paddle.nn.Layer):
    def __init__(self):
       super(kl_loss, self).__init__()
       self.cross_entropy_loss = paddle.nn.CrossEntropyLoss()

    def forward(self, p, q, label):
        ce_loss = 0.5 * (self.cross_entropy_loss(p, label) + self.cross_entropy_loss(q, label))
        kl_loss = self.compute_kl_loss(p, q)

        # carefully choose hyper-parameters
        loss = ce_loss + 0.3 * kl_loss

        return loss

    def compute_kl_loss(self, p, q):

        p_loss = F.kl_div(F.log_softmax(p, axis=-1), F.softmax(q, axis=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, axis=-1), F.softmax(p, axis=-1), reduction='none')

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2

        return loss
```

另外，我在复现论文时整理了一份方法论，希望能对大家有所帮助：
- [X2Paddle：手把手教你迁移代码——论文复现方法论](https://aistudio.baidu.com/aistudio/projectdetail/2276340)

# 作者简介

![](https://ai-studio-static-online.cdn.bcebos.com/6085edb36c944aca9b57a92eedf98d39aaf0ba9e25b44d7ab18984eb6804fc88)

本项目由[Mr.郑先生_](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/147378)和[七年期限](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/58637)共同完成
