# RMSProp

RMSProp 算法（Hinton，2012）修改 AdaGrad 以在非凸情况下表现更好，它改变梯度累积为指数加权的移动平均值，从而丢弃距离较远的历史梯度信息。RMSProp 与 Adadelta 的移动均值更新方式十分相似：

$$E[g^2]_{t}=0.9 E[g^2]_{t-1}+0.1 g_{t}^2$$

RMSProp参数更新公式如下，其中$\eta$是学习率， $g_{t}$是当前参数的梯度

$$\theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{E[g^2]_{t}+\epsilon}}g_{t}$$
RMSprop将学习速率除以梯度平方的指数衰减平均值。Hinton建议$\gamma$设置为0.9，默认学习率$\eta$为0.001