# Synchronized batch normalization(卡间同步批归一化)

## 1.Batch Normalization

### 1.1BN的定义

Batch Normalization(批标准化)，和普通的数据标准化类似，是将分散的数据统一的一种做法, 也是优化神经网络的一种方法。

<img src="https://paperswithcode.com/media/methods/batchnorm_1_UmYEcHj.png" alt="img" style="zoom:50%;" />

### 1.2BN的特点

就如普通的数据归一化一样，BN(*Batch Normalization*)能够控制输入的数据经过激活函数的敏感部分，从而避免梯度爆炸或消失。与普通的数据归一化不同的是，BN发生在隐层。

对于一个mini-batch的参数$ \mathcal{B} = {x_{1...m}} $

​			  可学习的参数为$\gamma, \beta$

<img src="https://github.com/FileCrasher/awesome-DeepLearning/blob/master/examples/JLU/images/diagram-20210824.png" alt="diagram-20210824" style="zoom:50%;" />

​																											图1

训练时的更新如图，其中输出：
$$
y_i = BN_{\gamma,\beta}(x_i)\tag{1}
$$

$$
\mu_\beta\leftarrow\frac1m\sum_{i=1}^mx_i\tag{2}
$$

$$
{\sigma_\beta}^2\leftarrow\frac1m\sum_{i=1}^m{(x_i - \mu_\beta)}^2\tag{3}
$$

$$
y_i\leftarrow\lambda\frac{x_i-\mu\beta}{\sqrt{{\sigma_\beta}^2+\epsilon}}\tag{4}
$$

当训练反向传播的时候，将更新**均值和方差对输入样本的梯度**[1]如 图1 右半部分。这也是训练模式和验证模式的主要区别。在验证时使用在训练集上学过来的方差$\sigma^2$和均值$\mu$，而不去更新它们。训练过程由上式更新方差$\sigma^2$和均值$\mu$，从而计算一个batch的normalization。

根据原论文[2]总结为如下四点：

- 通过Normalization修正输入的均值和方差，减少[内部协变量偏移](https://www.jianshu.com/p/a78470f521dd)从而加速训练
- 允许使用更高的学习率，但不用担心学习率发散
- 对模型起正则化的作用，减少Dropout的需求
- 可以使得数据变得非饱和非线性(saturating nonlinearities)从而避免过拟合

------



## 2.Synchronized batch normalization(SyncBN)

### 2.1为什么要跨卡同步 Batch Normalization

现有的标准 Batch Normalization 因为使用数据并行（Data Parallel），是单卡的实现模式，只在单个卡上对样本进行归一化，相当于减小了批量大小（batch-size）。 对于比较消耗显存的训练任务时，往往单卡上的相对批量过小，影响模型的收敛效果。跨卡同步 Batch Normalization 可以使用全局的样本进行归一化，这样相当于“增大”了批量大小，这样训练效果不再受到使用 GPU 数量的影响。

### 2.2 GPU 数量对批量大小的影响

深度学习平台在多卡 (GPU) 运算的时候都是采用的数据并行 (DataParallel) 。每次迭代，输入被等分成多份，然后分别在不同的卡上前向（forward）和后向（backward）运算，并且求出梯度，在迭代完成后合并梯度、更新参数，再进行下一次迭代。因为在前向和后向运算的时候，每个卡上的模型是单独运算的，所以相应的Batch Normalization 也是在卡内完成，所以实际BN所归一化的样本数量仅仅局限于卡内，相当于批量大小（batch-size）减小了。

### 2.3如何实现 SyncBN

SyncBN 的关键是在前向运算的时候拿到全局的均值和方差，在后向运算时候得到相应的全局梯度。最简单的实现方法是先同步求均值，再发回各卡然后同步求方差。

### 2.4算法论证
<img src="https://github.com/FileCrasher/awesome-DeepLearning/blob/master/examples/JLU/images/VGD0CHGAS_71F%7BBQ%24_5MV%24S.png" alt="VGD0CHGAS_71F%7BBQ%24_5MV%24S" style="zoom:50%;" />
假设有$ N $个输入样本 $ X = {x_1, ...x_N} $，那么方差可以表示为:
$$
\begin{aligned}
\sigma^2 = & \frac{\sum_{i=1}^N(x_i - \mu)^2}{N}\\
= & \frac{\sum_{i=1}^N{x_i}^2}{N} - \frac{(\sum_{i=1}^N{x_i})^2}{N^2}
\end{aligned}
\tag{5}
$$
其中$ \mu = \frac{\sum_{i=1}^Nx_i}{N} $.

首先计算每个GPU上的$ \sum{x_i} $和$ \sum{x_i}^2 $，使用$(5)$式得到全局的平均值和方差

然后计算每一个样本的归一化(normalization)
$$
y_i = \gamma\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} + \beta.
$$
用类似的方法可以计算$ \sum{x_i} $和$ \sum{x_i}^2 $在全局的梯度，用于反向传播(BP)

从公式中可以看出，与上文介绍的Batch Norm单卡操作不同的是，多卡同步的 BN计算的是所有卡的平均值和方差，变相增大了batch size。例如，假设原本四卡训练，丢进去数据后一张卡只有 2 的 batch size

1)按原来的算，只能拿到batch size为2的 normalization

2)按多卡同步算，能拿到 batch size 为$2 * 4 = 8%$的normalization，直观、客观上提升了batch size

## 3.两者的不同与共同点

a)相同点：计算方法相似；好处都是刚才归结的四点

b)不同点：比起传统的BN，多卡同步的BN计算时的参数是全局(所有卡一起的)，而传统BN只能算单卡的；多卡同步的BN *batch size*更大，使得多卡下模型更容易收敛，在多卡环境中更有优势。而且有BN的优点，用它！

## 4.Reference

> [1]Z.H. "MXNet Gluon上实现跨卡同步Batch Normalization." Amazon AI Applied Scientist: 1pp 24 Jul 2018<<http://www.baydue.com/news/writingskills/210.html>>
>
> [2]S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning, pages 448– 456, 2015. 2, 4, 5, 8, 9
>
> [3]Hang Zhang，Kristin Dana. Context Encoding for Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, 2, 9

