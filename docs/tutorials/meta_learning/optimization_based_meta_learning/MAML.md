# Optimization-Based Meta-Learning: MAML

MAML (Model-Agnostic Meta-Learning)： 与模型无关的元学习，可以适用于任何基于梯度优化的模型结构 (Base model architecture: 4 modules with a $3\times 3$ convolutions and 64 filters, followed by batch normalization, a ReLU nonlinearity, and $2\times 2$ max-pooling)。MAML 是典型的双层优化结构，其内层和外层的优化方式如下：

## MAML 内层优化方式

内层优化涉及到基学习器，从任务分布 $p(T)$ 中随机采样第 $i$ 个任务 $T_{i}$。任务 $T_{i}$ 上，基学习器的目标函数是：

$$ 
\min _{\phi} L_{T_{i}}\left(f_{\phi}\right) 
$$

其中，$f_{\phi}$ 是基学习器，$\phi$ 是基学习器参数，$L_{T_{i}}\left(f_{\phi}\right)$ 是基学习器在 $T_{i}$ 上的损失。更新基学习器参数：

$$
\theta_{i}^{N}=\theta_{i}^{N-1}-\alpha\left[\nabla_{\phi}
L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N-1}} 
$$

其中，$\theta$ 是元学习器提供给基学习器的参数初始值 $\phi=\theta$，在任务 $T_{i}$ 上更新 $N$ 后 $\phi=\theta_{i}^{N-1}$.

## MAML 外层优化方式

外层优化涉及到元学习器，将 $\theta_{i}^{N}$ 反馈给元学匀器，此时元目标函数是：

$$ 
\min _{\theta} \sum_{T_{i}\sim p(T)} L_{T_{i}}\left(f_{\theta_{i}^{N}}\right) 
$$

元目标函数是所有任务上验证集损失和。更新元学习器参数：

$$
\theta \leftarrow \theta-\beta \sum_{T_{i} \sim p(T)} \nabla_{\theta}\left[L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N}} 
$$

## MAML 算法流程

>1. randomly initialize $\theta$
>2. while not done do:
>   1. sample batch of tasks $T_i \sim p(T)$
>   2. for all $T_i$ do：
>       1. evaluate $\nabla_{\phi}L_{T_{i}}\left(f_{\phi}\right)$ with respect to K examples
>       2. compute adapted parameters with gradient descent: $\theta_{i}^{N}=\theta_{i}^{N-1}   -\alpha\left[\nabla_{\phi}L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N-1}} $
>   3. end for
>   4. update $\theta \leftarrow \theta-\beta \sum_{T_{i} \sim p(T)} \nabla_{\theta}\left[L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N}} $
>3. end while

MAML 中执行了两次梯度下降 (gradient by gradient)，分别作用在基学习器和元学习器上。图1给出了 MAML 中特定任务参数 $\theta_{i}^{*}$ 和元级参数 $\theta$ 的更新过程。

![MAML Schematic Diagram](../../../images/meta_learning/optimization_based_meta_learning/MAML/MAMLSchematicDiagram.png)
<center>
图1	MAML 示意图。灰色线表示特定任务所产生的梯度值（方向）；黑色线表示元级参数选择更新的方向（黑色线方向是几个特定任务产生方向的平均值）；虚线代表快速适应，不同的方向代表不同任务更新的方向。
</center>

## MAML 的优点

- 适用于任何基于梯度优化的模型结构。

- 双层优化结构，提升模型精度和泛化能力，避免过拟合。

## 对 MAML 的探讨

- 每个任务上的基学习器必须是一样的，对于差别很大的任务，最切合任务的基学习器可能会变化，那么就不能用 MAML 来解决这类问题。

- MAML 适用于所有基于随机梯度算法求解的基学习器，这意味着参数都是连续的，无法考虑离散的参数。对于差别较大的任务，往往需要更新网络结构。使用 MAML 无法完成这样的结构更新。

- MAML 使用的损失函数都是可求导的，这样才能使用随机梯度算法来快速优化求解，损失函数中不能有不可求导的奇异点，否则会导致优化求解不稳定。

- MAML 中考虑的新任务都是相似的任务，所以没有对任务进行分类，也没有计算任务之间的距离度量。对每一类任务单独更新其参数初始值，每一类任务的参数初始值不同，这些在 MAML 中都没有考虑。

## 参考文献

[1] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.](http://proceedings.mlr.press/v70/finn17a.html)





