# AdaDelta

由于AdaGrad单调递减的学习率变化过于激进，考虑一个改变二阶动量计算方法的策略：不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。这也就是AdaDelta名称中Delta的来历。Adadelta是 Adagrad 的扩展，旨在帮助缓解后者学习率单调下降的问题。

指数移动平均值大约就是过去一段时间的平均值，因此我们用这一方法来计算二阶累积动量：

$$E[g^2]_{t}=\gamma E[g^2]_{t-1}+(1-\gamma) g_{t}^2$$

其中$\gamma$类似于冲量，大约是0.9.现在将SGD更新的参数变化向量$\Delta \theta_{t}$:

$$\Delta \theta_{t}=-\eta \cdot g_{t,i}$$
$$\theta_{t+1}=\theta_{t}+\Delta \theta_{t}$$

在Adagrad中，$\Delta \theta_{t}$是由：

$$\Delta \theta_{t}=-\frac{\eta}{\sqrt{G_{t}+\epsilon}}\cdot g_{t,i}$$

表示的，现在用$E[g^2]_{t}$简单代替原来的对角矩阵$G_{t}$：

$$\Delta \theta_{t}=-\frac{\eta}{\sqrt{E[g^2]_{t}+\epsilon}}\cdot g_{t,i}$$

将分母简记为RMS，表示梯度的均方根误差：

$$\Delta \theta_{t}=-\frac{\eta}{RMS[g]_{t}}\cdot g_{t}$$

根据作者所说，更新中，定义指数衰减均值，代替梯度平方：

$$E[\Delta \theta^2]_{t}=\gamma E[\Delta \theta^2]_{t-1}+(1-\gamma)\Delta \theta_{t}^2$$

均方根误差变为：

$$RMS[\Delta \theta]_{t}=\sqrt{E[\Delta \theta^2]_{t}+\epsilon}$$

$RMS[\Delta \theta]_{t}$是未知的，我们近似用前一个时间步RMS值来估计：

$$\Delta \theta_{t}=-\frac{RMS[\Delta \theta]_{t-1}}{RMS[g]_{t}}g_{t}$$
$$\theta_{t+1}=\theta_{t}-\Delta \theta_{t}$$

Adadelta不用设置学习率，因为其更新规则已经把它消除了。

**优点**

+ 避免了二阶动量持续累积、导致训练过程提前结束的问题了
