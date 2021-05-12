# Nadam

Adam可以被看作是融合了RMSProp和momentum，RMSprop 贡献了历史平方梯度的指数衰减的平均值$v_{t}$，而动量则负责历史梯度的指数衰减平均值$m_{t}$，Nadam在Adam的基础上加入了一阶动量的累积,即Nesterov + Adam = Nadam，为了把NAG融入到Adam中，我们需要修改momentum的项$m_{t}$

momentum更新规则为：

$$g_{t}=\nabla_{\theta_{t}}J(\theta_{t})$$
$$m_{t}=\gamma m_{t-1}+\eta g_{t}$$
$$\theta_{t+1}=\theta_{t}-m_{t}$$

其中$\gamma$是动量的衰减项，$\eta$是步长，J是目标函数。将$m_{t}$代入上面的第三个式子展开得到：

$$\theta_{t+1}=\theta_{t}-(\gamma m_{t-1}+\eta g_{t})$$

动量包括在前面的动量向量方向上的一步和在当前梯度方向上的一步。

NAG允许我们在计算梯度之前通过动量步长更新参数，从而在梯度方向上执行更精确的步长。然后我们只需要更新梯度$g_{t}$来达到NAG：

$$g_{t}=\nabla_{\theta_{t}}J(\theta_{t}-\gamma m_{t-1})$$
$$m_{t}=\gamma m_{t-1}+\eta g_{t}$$
$$\theta_{t+1}=\theta_{t}-m_{t}$$

Dozat 提出按以下方式来修改 NAG ：与应用动量步骤两次不同的是：一次用来更新梯度$g_{t}$和一次用来更新参数$\theta_{t+1}$，直接对当前参数应用一个向前看的（look-ahead）动量向量：

$$g_{t}=\nabla_{\theta_{t}}J(\theta_{t})$$

$$m_{t}=\gamma m_{t-1}+\eta g_{t}$$

$$\theta_{t+1}=\theta_{t}-(\gamma m_{t}+\eta g_{t})$$

注意我们现在不再使用如上面展开的动量更新规则中的先前动量向量$m_{t-1}$，而是使用当前动量向量$m_{t}$来向前看,为了把Netsterov Momentum融入到Adam，我们把旧的动量向量用新的动量向量代替，Adam的更新规则为（注意不用修改$\hat v_{t}$）：

$$m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})g_{t}$$
$$\hat m_{t}=\frac{m_{t}}{1-\beta_{1}^t}$$
$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat v_{t}}+\epsilon}\hat m_{t}$$

上式子展开为：

$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat v_{t-1}}+\epsilon}(\frac{\beta_{1}m_{t-1}}{1-\beta_{1}^t}+\frac{(1-\beta_{1})g_{t}}{1-\beta_{1}^t})$$

$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat v_{t-1}}+\epsilon}(\beta_{1}\hat m_{t-1}+\frac{(1-\beta_{1})g_{t}}{1-\beta_{1}^t})$$

这个方程跟momentum的展开式类似，用$\hat m_{t-1}$替换$\hat m_{t-2}$，Nadam的更新规则为：

$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat v_{t}}+\epsilon}(\beta_{1}\hat m_{t}+\frac{(1-\beta_{1})g_{t}}{1-\beta_{1}^t})$$