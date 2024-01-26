三、MMoE多任务学习

1、概念：MMoE对于任务之间的关系进行明确地建模，并且学习任务特定的函数去平衡共享的表达。MMoE能够在不添加大量新的参数的情况下让参数进行自动分配以去捕捉shared task information以及task-specific information。

2、模型：

![image-20210724224303116](C:\Users\Emily\AppData\Roaming\Typora\typora-user-images\image-20210724224303116.png)

基于Shared-Bottom multi-task DNN structure（如上图( a ) (a)(a)）: Shared-Bottom: input -> bottom layers (shared) -> tower network (每个任务各自的)

如上图( c ) (c)(c)，MMoE有一组bottom networks, 每一个叫做一个expert, 在本文中， expert network是一个feed-forward network

然后为每个任务引入一个gating network。Gating networks 的输入是input features， 输出是softmax gates，即各个expert的权重

加权之后的expert结果被输入到task-specific 的tower networks

这样的话，不同任务的gating networks能够学到不同的专家混合方式，以此捕捉到任务之间的关系

MMoE更容易训练并且能够收敛到一个更好的loss，因为近来有研究发现modulation和gating机制能够提升训练非凸深度神经网络的可训练性。

3、作用：在推荐系统中，往往需要同时优化多个业务目标，承担起更多的业务收益

4、场景：电商场景：希望能够同时优化点击率和转换率，使得平台具备更加的目标；信息流场景，希望提高用户点击率的基础上提高用户关注，点赞，评论等行为，营造更好的社区氛围从而提高留存。

5、优缺点：

可以将每一个gate认为是weighted sum pooling操作。如果我们选择将gate换成max操作。x为输入，g(x)中分量最大值对应的expert被唯一选中，向上传递信号。如果g(x)与input无关，则模型退化成多个独立的NN模型stacking，这样就便于我们更方便理解模型的进化关系。 

此处MMoE是将MoE作为一个基本的组成单元，横向堆叠。也可以进行纵向堆叠，将上个MMoE的输出作为下一个输入。

如果任务相关度非常高，则OMoE和MMoE的效果近似，但是如果任务相关度很低，则OMoE的效果相对于MMoE明显下降，说明MMoE中的multi-gate的结构对于任务差异带来的冲突有一定的缓解作用。