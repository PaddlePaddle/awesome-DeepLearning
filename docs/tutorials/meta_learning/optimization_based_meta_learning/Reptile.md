# Optimization-Based Meta-Learning: Reptile

Reptil 可以适用于任何模型结构，是 MAML 的特例、近似和简化，主要解决 MAML 元学习器中出现的高阶导数问题。

在 MAML 的元学习器中，使用了求导数的算式来更新参数初始值，导致在计算中出现了任务损失函数的二阶导数。在 Reptile 的元学习器中，参数初始值更新时，直接使用了任务上的参数估计值和参数初始值之间的差，来近似损失函数对参数初始值的导数，进行参数初始值的更新，从而不会出现任务损失函数的二阶导数。

Peptile 有两个版本：Serial Version 和 Batched Version，两者的差异如下：

## Serial Version Reptile

单次更新的 Reptile，每次训练完一个任务的基学习器，就更新一次元学习器中的参数初始值。

(1) 任务上的基学习器记为 $f_{\phi}$ ，其中 $\phi$ 是基学习器中可训练的参数， $\theta$ 是元学习器提供给基学习器的参数初始值。在任务 $T_{i}$ 上，基学习器的损失函数是 $L_{T_{i}}\left(f_{\phi}\right)$ ，基学习器中的参数经过 $N$ 次迭代更新得到参数估计值：

$$
\theta_{i}^{N}=\operatorname{SGD}\left(L_{T_{i}}, {\theta}, {N}\right)
$$

(2) 更新元学习器中的参数初始值：

$$
\theta \leftarrow \theta+\varepsilon\left(\theta_{i}^{N}-\theta\right)
$$

**Serial Version Reptile 算法流程**

> 1. initialize $\theta$, the vector of initial parameters
> 2. for iteration=1, 2, ... do:
>       1. sample task $T_i$, corresponding to loss $L_{T_i}$ on weight vectors $\theta$
>       2. compute $\theta_{i}^{N}=\operatorname{SGD}\left(L_{T_{i}}, {\theta}, {N}\right)$
>       3. update $\theta \leftarrow \theta+\varepsilon\left(\theta_{i}^{N}-\theta\right)$
> 3. end for

## Batched Version Reptile

批次更新的 Reptile，每次训练完多个任务的基学习器之后，才更新一次元学习器中的参数初始值。

(1) 在多个任务上训练基学习器，每个任务从参数初始值开始，迭代更新 $N$ 次，得到参数估计值。

(2) 更新元学习器中的参数初始值：

$$
\theta \leftarrow \theta+\varepsilon \frac{1}{n} \sum_{i=1}^{n}\left(\theta_{i}^{N}-\theta\right)
$$

其中，$n$ 是指每次训练完 $n$ 个任务上的基础学习器后，才更新一次元学习器中的参数初始值。

**Batched Version Reptile 算法流程**

> 1. initialize $\theta$
> 2. for iteration=1, 2, ... do:
>       1. sample tasks $T_1$, $T_2$, ... , $T_n$,
>       2. for i=1, 2, ... , n do:
>            1. compute $\theta_{i}^{N}=\operatorname{SGD}\left(L_{T_{i}}, {\theta}, {N}\right)$
>       3. end for
>       4. update $\theta \leftarrow \theta+\varepsilon \frac{1}{n} \sum_{i=1}^{n}\left(\theta_{i}^{N}-\theta\right)$
> 3. end for

## 参考文献
[1] [Reptile: a Scalable Metalearning Algorithm](https://arxiv.org/abs/1803.02999v1)