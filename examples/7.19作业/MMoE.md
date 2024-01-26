# MMoE

## 概念

多任务模型通过学习不同任务的联系和差异，可提高每个任务的学习效率和质量。多任务学习的的框架广泛采用shared-bottom的结构，不同任务间共用底部的隐层。这种结构本质上可以减少过拟合的风险，但是效果上可能受到任务差异和数据分布带来的影响。也有一些其他结构，比如两个任务的参数不共用，但是通过对不同任务的参数增加L2范数的限制；也有一些对每个任务分别学习一套隐层然后学习所有隐层的组合。和shared-bottom结构相比，这些模型对增加了针对任务的特定参数，在任务差异会影响公共参数的情况下对最终效果有提升。缺点就是模型增加了参数量所以需要更大的数据量来训练模型，而且模型更复杂并不利于在真实生产环境中实际部署使用。因此提出了一个Multi-gate Mixture-of-Experts(MMoE)的多任务学习结构。MMoE模型刻画了任务相关性，基于共享表示来学习特定任务的函数，避免了明显增加参数的缺点。

## 模型

MMoE的模型如下

<img src="https://upload-images.jianshu.io/upload_images/3866322-89f028aed28ebba0.png?imageMogr2/auto-orient/strip|imageView2/2/w/1177/format/webp" alt="img" style="zoom:67%;" />



观察下图Shared Bottom的模型结构图和上图MMoE的结构，不难发现，MMoE实际上就是把Shared Bottom层替换成了一个双Gate的MoE层：

Shared Bottom to MoE 模型如下：

<img src="https://upload-images.jianshu.io/upload_images/3866322-62300d0cb1a621bb.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom: 67%;" />

我们先来看一下原始的Shared Bottom的方式，假设input为![x](https://math.jianshu.com/math?formula=x)共享的底层网络为![f(x)](https://math.jianshu.com/math?formula=f(x)), 然后将其输出喂到各任务独立输出层![h^k(x)](https://math.jianshu.com/math?formula=h%5Ek(x))，其中![k](https://math.jianshu.com/math?formula=k) 表示第![k](https://math.jianshu.com/math?formula=k) 个任务的独立输出单元，那么，第![k](https://math.jianshu.com/math?formula=k)个任务的输出![y^k](https://math.jianshu.com/math?formula=y%5Ek)即可表示为:
                                                                                                 ![y^k = h^k(f(x))](https://math.jianshu.com/math?formula=y%5Ek%20%3D%20h%5Ek(f(x)))
 而MoE共享层将这个大的**Shared Bottom网络**拆分成了多个小的**Expert网络**（如图所示，拆成了三个，并且保持参数个数不变，显然分成多少个Expert、每个多少参数，都是可以根据实际情况自己设定的）。我们把第![i](https://math.jianshu.com/math?formula=i)个Expert网络的运算记为![f_i(x)](https://math.jianshu.com/math?formula=f_i(x)),然后Gate操作记为![g(x)](https://math.jianshu.com/math?formula=g(x))，他是一个![n](https://math.jianshu.com/math?formula=n)元的softmax值（![n](https://math.jianshu.com/math?formula=n)是Expert的个数，有几个Expert，就有几元），之后就是常见的每个Expert输出的加权求和，假设MoE的输出为![y](https://math.jianshu.com/math?formula=y),那么可以表示为：
                                                                                              ![y = \sum_{i=1}^n g(x)_if_i(x)](https://math.jianshu.com/math?formula=y%20%3D%20%5Csum_%7Bi%3D1%7D%5En%20g(x)_if_i(x))
 如果只是这样的话，要完成多任务还得像Shared Bottom那样再外接不同的输出层，这样一搞似乎这个MoE层对多任务来说就没什么卵用了，因为它无法根据不同的任务来调整各个Expert的组合权重。所以论文的作者搞了多个Gate，每个任务使用自己独立的Gate，这样便从根源上，实现了网络参数会因为输入以及任务的不同都产生影响。
 于是，我们将上面MoE输出稍微改一下，用![g^k(x)](https://math.jianshu.com/math?formula=g%5Ek(x))表示第![k](https://math.jianshu.com/math?formula=k)个任务的们就得到了MMoE的输出表达：
                                                                                             ![y^k = \sum_{i=1}^ng^k(x)_if_i(x)](https://math.jianshu.com/math?formula=y%5Ek%20%3D%20%5Csum_%7Bi%3D1%7D%5Eng%5Ek(x)_if_i(x))



## 作用

在工业界基于神经网络的多任务学习在推荐等场景业务应用广泛，比如在推荐系统中对用户推荐物品时，不仅要推荐用户感兴趣的物品，还要尽可能地促进转化和购买，因此要对用户评分和购买两种目标同时建模。阿里之前提出的ESSM模型属于同时对点击率和转换率进行建模，提出的模型是典型的shared-bottom结构。多任务学习中有个问题就是如果子任务差异很大，往往导致多任务模型效果不佳。MMoE模型在多目标任务取得了不错的效果。



## 场景

推荐系统的多目标(ctr,互动率,转化率,etc.)

## 优缺点

优点：

避免了明显增加参数，MMoE中的multi-gate的结构对于任务差异带来的冲突有一定的缓解作用

**MMOE**是**MOE**的改进，相对于 **MOE**的结构中所有任务共享一个门控网络，**MMOE**的结构优化为每个任务都单独使用一个门控网络。这样的改进可以针对不同任务得到不同的 Experts 权重，从而实现对 Experts 的选择性利用，不同任务对应的门控网络可以学习到不同的Experts 组合模式，因此模型更容易捕捉到子任务间的相关性和差异性

缺点：

计算较繁琐
