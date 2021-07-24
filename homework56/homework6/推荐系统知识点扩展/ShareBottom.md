### ShareBottom

多目标建模目前业内有两种模式，一种叫Shared-Bottom模式，另一种叫MOE，MOE又包含MMOE和OMOE两种。MMOE也是Google提出的一套多目标学习算法结果，被应用到了Google的内部推荐系统中。

Shared-Bottom的思路就是多个目标底层共用一套共享layer，在这之上基于不同的目标构建不同的Tower。这样的好处就是底层的layer复用，减少计算量，同时也可以防止过拟合的情况出现。

Shared-Bottom 优点：降低overfit风险，利用任务之间的关联性使模型学习效果更强

Shared-Bottom 缺点：任务之间的相关性将严重影响模型效果。假如任务之间相关性较低，模型的效果相对会较差。