# 基于优化的元学习——MAML

## 1. 基础学习器和元学习器

### 1.1 基础学习器
基础学习器 (Base-Learner)，是基础层中的模型，每次训练基础学习器时，考虑的是单个任务上的数据集，其基本功能如下：
- 在单个任务上训练模型，学习任务的特性，找到规律，回答任务需要解决的问题。
- 从元学习器获取对于完成单个任务有帮助的经验，包括初始模型和初始参数等。
- 使用单个任务中的训练数据集，构建合适的目标函数，设计需要求解的优化问题，从初始模型和初始参数开始进行迭代更新。
- 在单个任务上训练完成后，将训练的模型和参数都反馈给元学习器。

### 1.2 元学习器
元学习器 (Meta-Learner)，是元层中的模型，对所有任务上的训练经验进行归纳总结。每次训练基础学习器后，元学习器都会综合新的经验，更新元学习器中的参数，其基本功能如下：
- 综合多个任务上基础学习器训练的结果。
- 对多个任务的共性进行归纳，在新任务上进行快速准确的推理，并且将推理输送给基础学习器，作为初始模型和初始参数值，或者是其他可以加速基础学习器训练的参数。
- 指引基础学习器的最优行为，指引基础学习器探索某个特定的新任务。
- 提取任务上与模型和训练相关的特征。

## 2. MAML中基础学习器的学习方式
从任务分布 $p(T)$ 中随机采样第 $i$ 个任务 $T_{i}$。任务 $T_{i}$ 上，基础学习器的目标函数是：

$$ 
\min _{\phi} L_{T_{i}}\left(f_{\phi}\right) 
$$

其中，$f_{\phi}$ 是基础学习器，$\phi$ 是基础学习器参数，$L_{T_{i}}\left(f_{\phi}\right)$ 是基础学习器在 $T_{i}$ 上的损失。更新基础学习器参数：

$$
\theta_{i}^{N}=\theta_{i}^{N-1}-\alpha\left[\nabla_{\phi}
L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N-1}} 
$$

其中，$\theta$ 是元学习器提供给基础学习器的参数初始值 $\phi=\theta$，在任务 $T_{i}$ 上更新 $N$ 后 $\phi=\theta_{i}^{N-1}$.

## 3. MAML中元学习器的学习方式
将 $\theta_{i}^{N}$ 反馈给元学匀器，此时元目标函数是：

$$ 
\min _{\theta} \sum_{T_{i}\sim p(T)} L_{T_{i}}\left(f_{\theta_{i}^{N}}\right) 
$$

元目标函数是所有任务上验证集损失和。更新元学习器参数：

$$
\theta \leftarrow \theta-\beta \sum_{T_{i} \sim p(T)} \nabla_{\theta}\left[L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N}} 
$$

## 4. MAML算法流程
>1. randomly initialize $\theta$
>2. while not done do:
>   1. sample batch of tasks $T_i \sim p(T)$
>   2. for all $T_i$ do：
>       1. evaluate $\nabla_{\phi}L_{T_{i}}\left(f_{\phi}\right)$ with respect to K examples
>       2. compute adapted parameters with gradient descent: $\theta_{i}^{N}=\theta_{i}^{N-1}   -\alpha\left[\nabla_{\phi}L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N-1}} $
>   3. end for
>   4. update $\theta \leftarrow \theta-\beta \sum_{T_{i} \sim p(T)} \nabla_{\theta}\left[L_{T_{i}}\left(f_{\phi}\right)\right]_{\phi=\theta_{i}^{N}} $
>3. end while

MAML中执行了两次梯度下降 (gradient by gradient)，分别作用在基学习器和元学习器上。图1给出了MAML中特定任务参数 $\theta_{i}^{*}$ 和元级参数 $\theta$ 的更新过程。

![](../../images/meta_learning/MAML/MAMLSchematicDiagram.png)
<center>
图1	MAML示意图。灰色线表示特定任务所产生的梯度值（方向）；黑色线表示元级参数选择更新的方向（黑色线方向是几个特定任务产生方向的平均值）；虚线代表快速适应，不同的方向代表不同任务更新的方向。
</center>

##  5. MAML的优点
- 适用于任何基于随机梯度下降法优化的基础学习器；
- 结构简单，可以很容易的和任何模型结构融合，提供元学习器和基础学习器的结构，加速梯度优化器的效果，提升模型精度和泛化能力，避免过拟合。

## 6. 对MAML的探讨
- 每个任务上的基础学习器必须是一样的，对于差别很大的任务，最切合任务的基础学习器可能会变化，那么就不能用MAML算法来解决这类问题。
- MAML算法适用于所有基于随机梯度算法求解的基础学习器，这意味着参数都是连续的，无法考虑离散的参数。对于差别较大的任务，往往需要更新网络结构。使用MAML算法无法完成这样的结构更新，而只能完成参数的快速准确更新，因此，只能适应差别较小的任务。
- MAML算法使用的损失函数都是可求导的，这样才能使用随机梯度算法来快速优化求解，损失函数中不能有不可求导的奇异点，否则会导致优化求解不稳定。
- MAML算法中考虑的新任务都是相似的任务，所以没有对任务进行分类，也没有计算任务之间的距离度量。对每一类任务单独更新其参数初始值，每一类任务的参数初始值不同，这些在 MAML 算法中都没有考虑。

## 参考文献
[1] [Model-Agnostic Meta-Learning for Fast Adaptation of Deep networks.](http://proceedings.mlr.press/v70/finn17a.html)





