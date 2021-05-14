# AdamW

L2 正则化是减少过拟合的经典方法，它会向损失函数添加由模型所有权重的平方和组成的惩罚项，并乘上特定的超参数以控制惩罚力度。加入L2正则以后，损失函数就变为：

$$L_{l_{2}}(\theta)=L(\theta)+1/2\gamma||\theta||^2$$

SGD就变为：

$$\theta_{t}=\theta_{t-1}-\nabla L_{l_{2}}(\theta_{t-1})=\theta_{t-1}-\nabla L(\theta_{t-1})-\gamma\theta_{t-1}$$

SGD+momentum就变为：

$$\theta_{t}=\theta_{t-1}-\gamma m_{t-1}-\eta(\nabla L(\theta_{t-1})+\gamma \theta_{t-1})$$
$$m_{t}=\gamma m_{t-1}+\eta(\nabla L(\theta_{t-1})+\gamma \theta_{t-1})$$
$$m_{t}=\gamma m_{t-1}+\eta(\nabla L{\theta_{t-1}})$$

最后一项是正则项产生。但是$m_{t}$的计算有上面两种，都可以。adamw的论文验证 $m_{t}=\gamma m_{t-1}+\eta(\nabla L{\theta_{t-1}})$ 效果好。

Adam就变为：

$$m_{t}=\gamma m_{t-1}+\eta(\nabla L(\theta_{t-1}))$$
$$v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})(\nabla L(\theta_{t-1})+\gamma \theta_{t-1})$$
AdamW最终的形式：

$$m_{t}=\beta_{1}m_{t-1}+(1-\beta_{1})\nabla L(\theta_{t-1})$$
$$v_{t}=\beta_{2}v_{t-1}+(1-\beta_{2})(\nabla L(\theta_{t-1}))^2$$
$$\theta_{t}=\theta_{t-1}-\eta(\frac{1}{\sqrt{\hat v_{t}}+\epsilon}\hat m_{t}-\gamma\theta_{t-1})$$

从上面的公式可以看出，AdamW本质上就是在损失函数里面加入了L2正则项，然后计算梯度和更新参数的时候都需要考虑这个正则项。AdamW使用在hugging face版的transformer中,BERT,XLNET,ELECTRA等主流的NLP模型，都是用了AdamW优化器

