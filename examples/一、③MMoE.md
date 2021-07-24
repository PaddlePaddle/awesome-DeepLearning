MMoE

概念

MMOE模型，全称为：Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts。

人们使用的大多数模型是一个模型完成一个任务的单任务模型，如果将多个任务混合在一起训练，可能出现参数之间互相干扰的情况二导致差于单独训练的效果。

当需要多任务混合训练时，为进行不相关任务的多任务学习，Google的MMOE模型应运而生。MMoE模型跳出了将整个隐藏层一股脑的共享的思维定式，而是将共享层有意识的（按照数据领域之类的）划分成了多个Expert，并引入了gate机制，得以个性化组合使用共享层。

模型

MMoE模型如图所示：

![v2-fd7ff14aeba3109c5f29fe07c478d013_720w](C:\Users\apple\Desktop\image\v2-fd7ff14aeba3109c5f29fe07c478d013_720w.png)

MMOE模型借鉴 MoE 的思路, 引入多个 Experts 网络, 然后再对每个 task 分别引入一个 gating network, gating 网络针对各自的 task 学习 experts 网络的不同组合模式, 即对 experts 网络的输出进行自适应加权. MMoE 网络可以形式化表达为:
![QQ截图20210722164840](C:\Users\apple\Desktop\image\QQ截图20210722164840.png)



作用

- Gate
  把输入通过一个线性变换映射到![nums_expert](https://math.jianshu.com/math?formula=nums_expert)维，再算个softmax得到每个Expert的权重
- Expert
  简单的基层全连接网络，relu激活，每个Expert独立权重

场景

大多数的场景没有多任务学习的必要，但是如果在如推荐系统等场景中，想要达成某些目标的时候，必须用到多任务学习。以给用户推荐视频为例，我们既希望提高用户的点击率，同时也希望提高视频的播放时长，视频点赞、转发等等... 这些目标的达成并非是简单的相辅相成，更多的可能是相互竞争的关系。要是我们只让模型学习点击率，那么经过训练的模型推荐结果很可能导致标题党和封面党大行其道，真正的好的视频却被雪藏了，这显然不是我们希望看到的。而如果一味的追求高点赞，也可能就忽略了一些相对冷门的或新的佳作。因此，我们无法追求某个单一目标的达成，而需要同时优化这些有利于产品良性循环的任务目标，让它们相互平衡，从而提升用户体验，带来和留住更多的用户。

优缺点

- 优点：解决传统的 multi-task 网络 (主要采用 Shared-Bottom Structure) 在任务相关性不强的情况下效果不佳的问题。
- 缺点：任务相关度非常高时，MMoE与传统算法效果不能拉开差距