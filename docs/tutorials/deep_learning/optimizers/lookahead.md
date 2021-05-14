# Lookahead

Lookahead是一种梯度下降优化器，它迭代的更新两个权重集合，"fast"和"slow"。直观地说，该算法通过向前看由另一个优化器生成的快速权值序列来选择搜索方向。
梯度下降的时候，走几步会退回来检查是否方向正确。避免突然掉入局部最低点。

Lookahead的算法描述如下：

1. 初始化参数$\phi_{0}$和目标函数L
2. 同步周期k,slow权重步长$alpha$和优化器A
	3. for t=1,2,...
	4. 同步参数$\theta_{t,0}=\phi_{t-1}$
	5. for i=1,2,...,k
		6. 采样一个minibatch的数据:$d \sim D$  
		7. $\theta_{t,i}=\theta_{t,i-1}+A(L,\theta_{t,i-1},d)$
	8. 外部更新$\phi_{t}=\phi_{t-1}+\alpha(\theta_{t,k}-\phi_{t-1})$
	返回参数

+ Fast weights

它是由内循环优化器（inner-loop）生成的k次序列权重；这里的优化器就是原有的优化器，如SGD，Adam等均可；其优化方法与原优化器并没有区别，例如给定优化器A，目标函数L，当前训练mini-batch样本d，这里会将该轮循环的k次权重，用序列都保存下来。

+ Slow Weights:

在每轮内循环结束后，根据本轮的k次权重，计算等到Slow Weights；这里采用的是指数移动平均（exponential moving average, EMA）算法来计算，最终模型使用的参数也是慢更新（Slow Weights）那一套，因此快更新（Fast Weights）相当于做了一系列实验，然后慢更新再根据实验结果选一个比较好的方向，这有点类似 Nesterov Momentum 的思想。
