**深度学习基础知识：**
**1.MMoE概念：**
MMoE(One-gate MoE mode)在一定程度上解决了上述问题，即虽然说多个模型还是共享相同的输入特征，但是每个任务都利用"gate network"来区分特征表达的权重，从而提高了模型的表达能力。但是这种模型架构的"区分"还不是很大，毕竟输入的特征还是只有一个，于是作者受集成学习（ensemble learning）思想的影响，提出了multi-experts，即我们可以把单个的共享特征看做是一个弱学习器的输入，那么，根据集成思想，若干弱学习器的组合可以作为一个强学习器来对结果进行推理，再通过"gate network"就可以极大地提高多任务模型的表达能力了。
![](https://ai-studio-static-online.cdn.bcebos.com/6d590fba615e4fc5b9cbcf7d796755a58352fb611f434d3da19efae8b72e467a)

**2.MMoE模型：**
MMoE实际上就是把Shared Bottom层替换成了一个双Gate的MoE层：
![](https://ai-studio-static-online.cdn.bcebos.com/f9e51b694bbc47bc900d0b8c13f9c4173ab07b039a3b4787a4d1c17cac198bb4)

先来看一下原始的Shared Bottom的方式，假设input为x共享的底层网络为f(x), 然后将其输出喂到各任务独立输出层h^k(x)，其中k 表示第k 个任务的独立输出单元，那么，第k个任务的输出y^k即可表示为:
![](https://ai-studio-static-online.cdn.bcebos.com/9c3ce153c51546948ee94a8bee0a14ca817d8430499f49a3a7c4ef9455d952da)

而MoE共享层将这个大的Shared Bottom网络拆分成了多个小的Expert网络（如图所示，拆成了三个，并且保持参数个数不变，显然分成多少个Expert、每个多少参数，都是可以根据实际情况自己设定的）。我们把第个Expert网络的运算记为,然后Gate操作记为，他是一个元的softmax值（是Expert的个数，有几个Expert，就有几元），之后就是常见的每个Expert输出的加权求和，假设MoE的输出为,那么可以表示为：
![](https://ai-studio-static-online.cdn.bcebos.com/d639a9653051443897c1d311f6d47c343be96b6e570a4657b1829d58b20cac4d)

如果只是这样的话，要完成多任务还得像Shared Bottom那样再外接不同的输出层，这样一搞似乎这个MoE层对多任务来说就没什么卵用了，因为它无法根据不同的任务来调整各个Expert的组合权重。所以论文的作者搞了多个Gate，每个任务使用自己独立的Gate，这样便从根源上，实现了网络参数会因为输入以及任务的不同都产生影响。
于是，我们将上面MoE输出稍微改一下，用表示第个任务的们就得到了MMoE的输出表达：
![](https://ai-studio-static-online.cdn.bcebos.com/a7e2ed4963eb4b8bb40065d18037a207bf090322f37f435c9240da4375b138d8)

其中，
Gate：把输入通过一个线性变换映射到nums_expert维，再算个softmax得到每个Expert的权重；
Expert：简单的基层全连接网络，relu激活，每个Expert独立权重、

**3.MMoE作用：**
在MMoE提出之前，多任务模型已经有许多经典架构被提出，其中绝大多数的优化都基于share-bottom架构，即不同的任务共享相同的feature或feature_map。
![](https://ai-studio-static-online.cdn.bcebos.com/c676dc5a339a4cf98c10b86a9f74f73c73e4c774f8d24d3ea11d3683d4650bce)
然而，这种架构极大地限制了模型表达的能力，为什么这么说？因为我们在共享特征的上层直接接入了多个目标的输出，而由于多个任务各自有不同的数据分布，也就是说我们对不同任务的输出具有一定的差异性，而相同的特征输入会极大地削弱模型的多任务输出表达而在某种程度上降低了多目标模型的泛化能力。
![](https://ai-studio-static-online.cdn.bcebos.com/9997919f53724b5ebcd2727ad87f01559305fc1cb2fe464d9b9ceebe5b4c640f)
那么，如何去降低这种架构带来的影响，作者首先提出了One-gate MoE model。

**4.MMoE场景：**
随着深度学习的不断发展，其在推荐场景下的应用也越来越广泛了。在业界研发人员不断追求模型CTR效果的同时，深度学习模型复杂度也在急剧上升，然而由于推荐业务场景本身的低响应要求，导致我们希望在为线上用户推荐TA喜欢的物品的同时，还要保证模型足够简洁，以达到线上算力的要求。一般来说，大部分人见过更多的都是单任务的模型，即一个模型完成一个任务。如果需要完成多个任务的话，则可以针对每个任务单独训练一个模型，像这样：
![](https://ai-studio-static-online.cdn.bcebos.com/d8b6d5b687024b798f642ba458246d5802dce2d1d9d546c1aafe7fd7820ffd46)
似乎完全没有要将多个任务混在一起训练的必要，因为这样极有可能导致参数相互干扰而差于单独训练的效果。大多数的场景确实没有多任务学习的必要，但是如果你做过推荐系统，就会发现在想要达成某些目标的时候，非得多任务一起上不可。

**5.MMoE优缺点：**
 优点：改善了RNN中存在的长期依赖问题；LSTM的表现通常比时间递归神经网络及隐马尔科夫模型（HMM）更好；作为非线性模型，LSTM可作为复杂的非线性单元用于构造更大型深度神经网络。

缺点：一个缺点是RNN的梯度问题在LSTM及其变种里面得到了一定程度的解决，但还是不够。它可以处理100个量级的序列，而对于1000个量级，或者更长的序列则依然会显得很棘手；另一个缺点是每一个LSTM的cell里面都意味着有4个全连接层(MLP)，如果LSTM的时间跨度很大，并且网络又很深，这个计算量会很大，很耗时。
