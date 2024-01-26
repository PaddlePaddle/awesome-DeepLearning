# 四、ShareBottom多任务学习知识点补充
 share bottom网络
  将id特征embedding，和dense特征concat一起，作为share bottom网络输入，id
![](https://ai-studio-static-online.cdn.bcebos.com/cef5ac4715a84184ad6aab1d92fc27782fa3d8bc77f24b9ab35484921b716ceb)

  特征embedding可以使用end2end和预训练两种方式。预训练可以使用word2vec，GraphSAGE等工业界落地的算法，训练全站id embedding特征，在训练dnn或则multi task的过程中fine-tune。end2end训练简单，可以很快就将模型train起来，直接输入id特征，模型从头开始学习id的embedding向量。这种训练方式最致命的缺陷就是训练不充分，某些id在训练集中出现次数较少甚至没有出现过，在inference阶段遇到训练过程中没有遇到的id，就直接走冷启了。这在全站item变化比较快的情况下，这种方式就不是首选的方式。
  多任务学习人工调的参数相对DNN较多，如上图所示，我们有三个损失，这三个损失怎么训练？三个子任务的输出，排序时如何使用？正负样本不均衡时，改如何处理？这三个问题分别对应了loss weight，output weight ，label weight。下面针对这三个问题，依次展开。

1、 label weight
  调整正样本的比例，对于正负样本分布不均衡的时候，该参数可以使得模型提高对正样本的关注，提高模型的召回,auc指标。
![](https://ai-studio-static-online.cdn.bcebos.com/f3dd8eb645b243729428d4d5d3d275b43229039ac25141a9ae247169ec676d34)

点击AUC(绿色对点击正样本加权)
![](https://ai-studio-static-online.cdn.bcebos.com/8c6500046d554730a67fda12258fa798cd1a9c6a98c74fb1b54a5fb039b8577a)
转化AUC(绿色对订单正样本加权)
  从上面两个图可以看出，当正负样本比例相差悬殊时(我们点击正负样本比例是1：10，订单正负样本比例是千分之一的比例)，提高正样本权重，对AUC提升会比较明显，订单AUC得到明显的提升。每个子任务正负样本比都为1:1可能导致各个子任务之间学习相互抑制。对于正负样本比例比较悬殊的，可以调整正样本为负样本的一半，效果比较好。
2、 loss weight
  调整损失的权重， 让共享层更关注某一个任务，也能解决一部分样本分布不均衡带来的过拟合。通常，MTL训练的过程中，是对多个子任务的损失线性加权:
  这样有个明显的缺点，就是这个wi
 需要很强的先验知识，人工预设，在训练的过程中保持不变。大家都知道，损失值在深度学习中的地位，直接决定了梯度的大小和传播，调参给出的权值很难保证我们给出的就是最优解，使得每个子任务都达到最优。这也是多任务学习一个主流的研究方向—pareto 最优解。这是一个经济学的概念，感兴趣的同学可以去了解一下，我会在下面附上相关的参考文献。言归正传，loss weight不合适，会导致正负样本严重不均衡的子任务会快出现过拟合的情况，如下图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/7dd83d150a464ad9ad14f0e3d1beb6d1bef9f5524d714cbea790137139f32732)
订单AUC
  深度学习的表达能力很强，模型很快就学习到了正样本的特征，继续训练导致模型出现过拟合的情况。而对于正样本较多的子任务，模型还没有学习充分。因此，我们需要给损失一个合适的权重，某种程度上也是在调整每个子任务的学习率。在给定的权值下，每个子任务达到局部最优解。
3、 output weight
  调整模型输出的权重组合。不同的权重值，影响排序时对应的子任务的重要性。对于输出的权重组合，可以利用grid search的思想，选出我们关心的离线评估指标最高的一个组合作为线上排序依据，在离线空间的组合逼近连续空间的最优值。最直接的做法可以将三个子任务的输出，在0到1区间，每隔0.1采样一个权值，在这1000组参数中，选择离线指标最高的一个组合作为最终的输出加权
下面是伪代码：


```python
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

```
