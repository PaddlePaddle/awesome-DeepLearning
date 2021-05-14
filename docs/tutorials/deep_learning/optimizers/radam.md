# RAdam

RAdam（Rectified Adam）是Adam优化器的一个变体，它引入了一项来纠正自适应学习率的方差，试图解决Adam的收敛性差的问题。作者认为收敛性差的原因是自适应学习率在早期的模型训练的过程中有较大的方差，这是因为训练使用的数据有限。为了减小方差，在训练前几个epoch的时候使用一个较小的学习率，这证明了热身启发式（warmup）的合理性。warmup的方式启发了RAdam来纠正方差问题。
$$g_{t}=\nabla_{\theta}f_{t}(\theta_{t-1})$$
$$v_{t}=\frac{1}{\beta_{2}}v_{t-1}+(1-\beta_{2})g_{t}^2$$
$$m_{t}=\beta_{1} m_{t-1}+(1-\beta_{1})g_{t}$$
$$\hat m_{t}=\frac{m_{t}}{1-\beta_{1}^t}$$

$$\rho_{t}=\rho_{\infty}-\frac{2t\beta_{2}^2}{1-\beta_{2}^t}$$
$$\rho_{\infty}=\frac{2}{1-\beta_{2}}-1$$

其中$m_{t}$是一阶矩（动量），$v_{t}$是二阶矩(自适应学习率),$\eta$是学习率。

当$\rho_{t} > 4$的时候：
自适应的学习率的计算公式为：

$$l_{t}=\sqrt{(1-\beta_{2}^t)/v_{t}}$$

方差矫正项计算公式为：

$$r_{t}=\sqrt{\frac{(\rho_{t}-4)(\rho_{t}-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2)\rho_{t}}}$$

我们使用自适应的momentum方法来更新参数

$$\theta_{t}=\theta_{t-1}-\alpha_{t} r_{t}\hat m_{t} l_{t}$$

如果方差不容易得到（tractable），我们采用下面的公式：

$$\theta_{t}=\theta_{t-1}-\alpha_{t} \hat m_{t}$$

