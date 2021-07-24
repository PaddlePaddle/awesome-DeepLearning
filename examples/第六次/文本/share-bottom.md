**深度学习基础知识：**
**1.Shared-Bottom概念：**
多目标建模目前业内有两种模式，一种叫Shared-Bottom模式，另一种叫MOE，MOE又包含MMOE和OMOE两种。MMOE也是Google提出的一套多目标学习算法结果，被应用到了Google的内部推荐系统中。如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/bc6f3c51a2e74273b4dc82a665ebae0471db4d911b164f35bcdc61167402d0eb)
Shared-Bottom的思路就是多个目标底层共用一套共享layer，在这之上基于不同的目标构建不同的Tower。这样的好处就是底层的layer复用，减少计算量，同时也可以防止过拟合的情况出现。

**2.Shared-Bottom模型：**
将id特征embedding，和dense特征concat一起，作为share bottom网络输入，id
训练和测试样本的处理 为了能够提高模型的训练效率和预测准确率，Youtube采取了诸多处理训练样本的工程措施，主要有3点：
![](https://ai-studio-static-online.cdn.bcebos.com/bd78a4856c884fec826b317118c461d9d40223122a4f4afe8cad2523cb020960)

特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。这在全站item变化比较快的情况下，这种方式就不是首选的方式。
  多任务学习人工调的参数相对DNN较多，如上图所示，我们有三个损失，这三个损失怎么训练？三个子任务的输出，排序时如何使用？正负样本不均衡时，改如何处理？这三个问题分别对应了loss weight，output weight ，label weight。
（1）label weight
  调整正样本的比例，对于正负样本分布不均衡的时候，该参数可以使得模型提高对正样本的关注，提高模型的召回,auc指标。
  ![](https://ai-studio-static-online.cdn.bcebos.com/16f138661cf348e7b8139e91fabd4b127c98d2a2732d43c5b6c417619b791089)
  ![](https://ai-studio-static-online.cdn.bcebos.com/a6601a47ce484239a8afec1ac46ca34a5d2582a6ab8b4101ad6ca67b9d0c8a0b)
  从上面两个图可以看出，当正负样本比例相差悬殊时(我们点击正负样本比例是1：10，订单正负样本比例是千分之一的比例)，提高正样本权重，对AUC提升会比较明显，订单AUC得到明显的提升。每个子任务正负样本比都为1:1可能导致各个子任务之间学习相互抑制。对于正负样本比例比较悬殊的，可以调整正样本为负样本的一半，效果比较好。
 （2）loss weight
  调整损失的权重， 让共享层更关注某一个任务，也能解决一部分样本分布不均衡带来的过拟合。通常，MTL训练的过程中，是对多个子任务的损失线性加权:
  ![](https://ai-studio-static-online.cdn.bcebos.com/cd41d21a0a0f459b84ad1fa2a3585d173dfed36b06924d4eb17fdc2110f5cc8d)
   这样产生了个问题，权重的确定需要很强的先验知识，人工预设，在训练的过程中保持不变。大家都知道，损失值在深度学习中的地位，直接决定了梯度的大小和传播，调参给出的权值很难保证我们给出的就是最优解，使得每个子任务都达到最优。这也是多任务学习一个主流的研究方向—pareto 最优解。这是一个经济学的概念，感兴趣的同学可以去了解一下，我会在下面附上相关的参考文献。言归正传，loss weight不合适，会导致正负样本严重不均衡的子任务会快出现过拟合的情况，如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/61d6bbb6c0114428be747694539928e7911c8aa95de24bfc9bd66c7d0a85dc29)
深度学习的表达能力很强，模型很快就学习到了正样本的特征，继续训练导致模型出现过拟合的情况。而对于正样本较多的子任务，模型还没有学习充分。因此，我们需要给损失一个合适的权重，某种程度上也是在调整每个子任务的学习率。在给定的权值下，每个子任务达到局部最优解。
（3）output weight
  调整模型输出的权重组合。不同的权重值，影响排序时对应的子任务的重要性。对于输出的权重组合，可以利用grid search的思想，选出我们关心的离线评估指标最高的一个组合作为线上排序依据，在离线空间的组合逼近连续空间的最优值。最直接的做法可以将三个子任务的输出，在0到1区间，每隔0.1采样一个权值，在这1000组参数中，选择离线指标最高的一个组合作为最终的输出加权，下面是伪代码：
best_offline_score = 0
best_weight = None
for i in range(0,1.1,0.1):
	for j in range(0,1.1,0.1):
		for k in range(0,1.1,0.1):
			score = task1_score*i+task2_score*j+task3_score * k
			current_metric = calculate_offline_metrics(socre)
			if current_metric > best_offline_score:
				best_offline_score = current_metric
				best_weight = [i, j, k] #记录最优的权值组合		

**3.Shared-Bottom作用：**
独立的单任务处理，忽略了问题之间所富含的丰富的关联信息。MTL可以找到各目标优化时的trade off。 比如Single Task 在优化转换的时候，会对点击带来负面效果，MTL可以降低甚至消除这个负面效果。常用多任务有hard和soft两种模式，
  ![](https://ai-studio-static-online.cdn.bcebos.com/ccb64bece5294b899352f0f127acfbdb26685e734feb4f08b45e76a674578c88)
 共享参数得过拟合几率比较低，hard parameter网络每个任务有自己的参数，模型参数之间的距离会作为正则项来保证参数尽可能相似。hard模式对于关联性比较强的任务，降低网络过拟合，提升了泛化效果。常用的是share bottom 的网络 可以共享参数，提高泛化（improving generalization）。
 
**4.Shared-Bottom场景：**
  多任务学习(Multi-task learning)在cv和nlp领域已经得到广泛的应用，无论是经典的maskrcnn—同时预测bounding box的位置和类别，还是称霸nlp的bert—预测某个单词和句子是否相关联，都属于多任务模型。在推荐中是基于隐式反馈来进行推荐的，用户对于推荐结果是否满意通常依赖很多指标(点击，收藏，评论，购买等)，因此在排序中，我们需要综合考虑多个目标，尽可能使所有目标都达到最优。多任务学习是解决多目标排序问题的方案之一。
  比如我们想优化订单，会给予订单样本比较大的权值，模型在学习过程中，会将重点放在订单部分，因为这部分会引起比较大的损失值，忽略点击样本。这样的情况下，会导致模型过于关注某一部分，模型学到的偏离整体样本分布，线上相当于用子空间去预测全空间的分布。
  
**5.Shared-Bottom优缺点：**
Shared-Bottom 优点：降低overfit风险，利用任务之间的关联性使模型学习效果更强

Shared-Bottom 缺点：任务之间的相关性将严重影响模型效果。假如任务之间相关性较低，模型的效果相对会较差。

