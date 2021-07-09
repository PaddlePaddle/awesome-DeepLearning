# 深度学习的发展历史

- **1940年代**：首次提出神经元的结构，但权重是不可学的。
- **50-60年代**：提出权重学习理论，神经元结构趋于完善，开启了神经网络的第一个黄金时代。
- **1969年**：提出异或问题（人们惊讶的发现神经网络模型连简单的异或问题也无法解决，对其的期望从云端跌落到谷底），神经网络模型进入了被束之高阁的黑暗时代。
- **90年代左右**：1986年新提出的适用于多层感知器（MLP）的BP算法解决了异或问题。但由于其存在梯度消失问题，且随着90年代后理论更完备并且实践效果更好的SVM等机器学习模型的兴起，神经网络并未得到重视。
![8212741e9a70495ea467a2d2a861baff9ecd964aa67447e8a415b162dd5d01a4](.\images\8212741e9a70495ea467a2d2a861baff9ecd964aa67447e8a415b162dd5d01a4.png)
<center>深度学习的发展历程</center>
- **发展期 2006年 - 2012年**
  2006年提出了深层网络训练中梯度消失问题的解决方案：无监督预训练对权值进行初始化+有监督训练微调。至此开启了深度学习在学术界和工业界的浪潮。
  2011年，ReLU激活函数被提出，该激活函数能够有效的抑制梯度消失问题。
- **爆发期 2012 - 2017**
2012年CNN网络AlexNet一举夺得ImageNet图像识别比赛冠军，且碾压第二名（SVM方法）的分类性能。CNN开始吸引到了众多研究者的注意。
# 人工智能、机器学习、深度学习有什么区别和联系
概括来说，人工智能、机器学习和深度学习覆盖的技术范畴是逐层递减的。人工智能是最宽泛的概念。机器学习是当前比较有效的一种实现人工智能的方式。深度学习是机器学习算法中最热门的一个分支，近些年取得了显著的进展，并替代了大多数传统机器学习算法。三者的关系如 图1 所示，即：人工智能 > 机器学习 > 深度学习。

<img src=".\images\5521d1d951c440eb8511f03a0b9028bd63357aec52e94189b5ab3f55d63369d7.png" alt="5521d1d951c440eb8511f03a0b9028bd63357aec52e94189b5ab3f55d63369d7" style="zoom:48%;" />

<center>图2：人工智能、机器学习和深度学习三者关系示意</center>
如字面含义，人工智能是研发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。由于这个定义只阐述了目标，而没有限定方法，因此实现人工智能存在的诸多方法和分支，导致其变成一个“大杂烩”式的学科。
# 神经元、单层感知机、多层感知机
- **神经元**：
  由人类神经元抽象出一个类神经元的运算模型，称为M-P模型。除去延迟等细枝末梢，简化后的模型图所示：<img src=".\images\v2-89a2f7f5e235517c250aa9c814fe413a_720w.jpg" alt="v2-89a2f7f5e235517c250aa9c814fe413a_720w" style="zoom:80%;" />
  其中I为输入，相当突触。W为权重，跟细胞体有关，T为阈值函数。I经过W加权累加后送到T，如果累加和大于T的阈值，则神经元被激活，否则不激活。
  
- **单层感知器**：
  M-P模型抽象出了神经元的结构，但并没有给出W权重该如何确定，是强还是弱，什么时候强，什么时候弱，由什么因素决定。Hebb学习规则尝试着给了回答。结合M-P模型和Hebb学习规则，罗森布拉特发明了感知机。最大的贡献是解决了M-P模型中参数W的学习问题。
  单层感知器是创建的第一个提出的神经模型。 神经元局部记忆的内容由权重向量组成。 单层感知器的计算是在输入向量的总和的计算上进行的，每个输入向量的值乘以权重向量的相应元素。

- **多层感知机**：
  多层感知机（MLP）就是含有至少一个 隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出都通过激活函数进行变换。MLP并没有规定隐层的数量，因此可以根据各自的需求选择合适的隐层层数。且对于输出层神经元的个数也没有限制。MLP神经网络结构模型如下：

<img src=".\images\749674-1f47a199a6ce5008.webp" alt="749674-1f47a199a6ce5008" style="zoom: 67%;" />

<center>多层感知机</center>
# 什么是前向传播

<img src="https:////upload-images.jianshu.io/upload_images/15086835-4264771413cbeb4d.png?imageMogr2/auto-orient/strip|imageView2/2/w/1070/format/webp" alt="img" style="zoom: 45%;" />

<center>三层DNN，输入层-隐藏层-输出层</center>

对于第2层第1个节点的输出![a_1^2](https://math.jianshu.com/math?formula=a_1%5E2)有：![a_1^2=\sigma(z_1^2) = \sigma(w_{11}^2x_1 + w_{12}^2x_2 + w_{13}^2x_3 + b_1^{2})](https://math.jianshu.com/math?formula=a_1%5E2%3D%5Csigma(z_1%5E2)%20%3D%20%5Csigma(w_%7B11%7D%5E2x_1%20%2B%20w_%7B12%7D%5E2x_2%20%2B%20w_%7B13%7D%5E2x_3%20%2B%20b_1%5E%7B2%7D))

对于第3层第1个节点的输出![a_1^3](https://math.jianshu.com/math?formula=a_1%5E3)有：![a_1^3=\sigma(z_1^3) = \sigma(w_{11}^3a_1^2 + w_{12}^3a_2^2 + w_{13}^3a_3^2 + b_1^{3})](https://math.jianshu.com/math?formula=a_1%5E3%3D%5Csigma(z_1%5E3)%20%3D%20%5Csigma(w_%7B11%7D%5E3a_1%5E2%20%2B%20w_%7B12%7D%5E3a_2%5E2%20%2B%20w_%7B13%7D%5E3a_3%5E2%20%2B%20b_1%5E%7B3%7D))

一般化的，假设l-1层有m个神经元，对于![a_j^l](https://math.jianshu.com/math?formula=a_j%5El)有：

![a_j^l=\sigma(z_j^l)=\sigma(\sum_{k=1}^m w_{jk}^la_k^{l-1}+b_j^l)](https://math.jianshu.com/math?formula=a_j%5El%3D%5Csigma(z_j%5El)%3D%5Csigma(%5Csum_%7Bk%3D1%7D%5Em%20w_%7Bjk%7D%5Ela_k%5E%7Bl-1%7D%2Bb_j%5El))

也就是第l层第j个神经元的输入为与它相连的上一层每个神经元的输出加权求和后加上该神经元对应的偏置，该神经元所做的工作只是把这个结果做一个非线性激活。

# 什么是反向传播

当通过前向传播得到由任意一组随机参数W和b计算出的网络预测结果后，我们可以利用损失函数相对于每个参数的梯度来对他们进行修正。事实上神经网络的训练就是这样一个不停的前向-反向传播的过程，直到网络的预测能力达到我们的预期。

假设选择最简单的均方误差和作为损失函数：![J(W,b,x,y)=\frac {1}{2}||a^L-y||_2^2](https://math.jianshu.com/math?formula=J(W%2Cb%2Cx%2Cy)%3D%5Cfrac%20%7B1%7D%7B2%7D%7C%7Ca%5EL-y%7C%7C_2%5E2)

下面就根据这个损失函数更新每一层的w,b

根据前向传播的公式，输出层L的输出![a^L=\sigma(z^L)=\sigma(w^La^{L-1}+b^L)](https://math.jianshu.com/math?formula=a%5EL%3D%5Csigma(z%5EL)%3D%5Csigma(w%5ELa%5E%7BL-1%7D%2Bb%5EL))

带入到损失函数中，有![J(W,b,x,y)=\frac {1}{2}||\sigma (w^La^{L-1}+b^L)-y||_2^2](https://math.jianshu.com/math?formula=J(W%2Cb%2Cx%2Cy)%3D%5Cfrac%20%7B1%7D%7B2%7D%7C%7C%5Csigma%20(w%5ELa%5E%7BL-1%7D%2Bb%5EL)-y%7C%7C_2%5E2)

根据复合函数链式求导法则，L层参数![W^L,b^L](https://math.jianshu.com/math?formula=W%5EL%2Cb%5EL)的梯度容易求得：

![\delta w^L=\frac{\partial J(W,b,x,y)}{\partial W^L}=\frac {\partial J(W,b,x,y)}{\partial z^L}\cdot \frac{\partial z^L}{\partial W^L}=(a^L-y)\odot \sigma^{(1)}(z^L)(a^{L-1})^T](https://math.jianshu.com/math?formula=%5Cdelta%20w%5EL%3D%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20W%5EL%7D%3D%5Cfrac%20%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5EL%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20z%5EL%7D%7B%5Cpartial%20W%5EL%7D%3D(a%5EL-y)%5Codot%20%5Csigma%5E%7B(1)%7D(z%5EL)(a%5E%7BL-1%7D)%5ET)

![\delta b^L=\frac{\partial J(W,b,x,y)}{\partial b^L} = \frac{\partial J(W,b,x,y)}{\partial z^L}\frac{\partial z^L}{\partial b^L} =(a^L-y)\odot \sigma^{(1)}(z^L)](https://math.jianshu.com/math?formula=%5Cdelta%20b%5EL%3D%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20b%5EL%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5EL%7D%5Cfrac%7B%5Cpartial%20z%5EL%7D%7B%5Cpartial%20b%5EL%7D%20%3D(a%5EL-y)%5Codot%20%5Csigma%5E%7B(1)%7D(z%5EL))

显然，两式有一部分是重叠的，将这部分记做![\delta^L](https://math.jianshu.com/math?formula=%5Cdelta%5EL)，![\delta^L=\frac{\partial J(W,b,x,y)}{\partial b^L} =(a^L-y)\odot \sigma^{(1)}(z^L)](https://math.jianshu.com/math?formula=%5Cdelta%5EL%3D%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20b%5EL%7D%20%3D(a%5EL-y)%5Codot%20%5Csigma%5E%7B(1)%7D(z%5EL))

这一规律同样适用在非输出层的隐藏层L-1,L-2,...l,...1，我们只需要求出损失函数相对l层非激活输出![z^l](https://math.jianshu.com/math?formula=z%5El)的导数，再根据前向传播公式![z^l=w^la^{l-1}+b^l](https://math.jianshu.com/math?formula=z%5El%3Dw%5Ela%5E%7Bl-1%7D%2Bb%5El)便可以轻易的求得![W^L,b^l](https://math.jianshu.com/math?formula=W%5EL%2Cb%5El)

同样，根据链式求导法则，

![\delta^l =\frac{\partial J(W,b,x,y)}{\partial z^l} = \frac{\partial J(W,b,x,y)}{\partial z^L}\frac{\partial z^L}{\partial z^{L-1}}\frac{\partial z^{L-1}}{\partial z^{L-2}}...\frac{\partial z^{l+1}}{\partial z^{l}}](https://math.jianshu.com/math?formula=%5Cdelta%5El%20%3D%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5El%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5EL%7D%5Cfrac%7B%5Cpartial%20z%5EL%7D%7B%5Cpartial%20z%5E%7BL-1%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7BL-1%7D%7D%7B%5Cpartial%20z%5E%7BL-2%7D%7D...%5Cfrac%7B%5Cpartial%20z%5E%7Bl%2B1%7D%7D%7B%5Cpartial%20z%5E%7Bl%7D%7D)

![\frac{\partial J(W,b,x,y)}{\partial W^l} = \frac{\partial J(W,b,x,y)}{\partial z^l} \frac{\partial z^l}{\partial W^l} = \delta^{l}(a^{l-1})^T](https://math.jianshu.com/math?formula=%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20W%5El%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5El%7D%20%5Cfrac%7B%5Cpartial%20z%5El%7D%7B%5Cpartial%20W%5El%7D%20%3D%20%5Cdelta%5E%7Bl%7D(a%5E%7Bl-1%7D)%5ET)

至此，问题的关键转化成如何求解![\delta^l](https://math.jianshu.com/math?formula=%5Cdelta%5El)，既然是反向传播，在求第l层参数时，![\delta^L,...,\delta^{l+1}](https://math.jianshu.com/math?formula=%5Cdelta%5EL%2C...%2C%5Cdelta%5E%7Bl%2B1%7D)都是已知的，还是根据链式求导法则：

![\delta^{l} = \frac{\partial J(W,b,x,y)}{\partial z^l} = \frac{\partial J(W,b,x,y)}{\partial z^{l+1}}\frac{\partial z^{l+1}}{\partial z^{l}} = \delta^{l+1}\frac{\partial z^{l+1}}{\partial z^{l}}](https://math.jianshu.com/math?formula=%5Cdelta%5E%7Bl%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5El%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J(W%2Cb%2Cx%2Cy)%7D%7B%5Cpartial%20z%5E%7Bl%2B1%7D%7D%5Cfrac%7B%5Cpartial%20z%5E%7Bl%2B1%7D%7D%7B%5Cpartial%20z%5E%7Bl%7D%7D%20%3D%20%5Cdelta%5E%7Bl%2B1%7D%5Cfrac%7B%5Cpartial%20z%5E%7Bl%2B1%7D%7D%7B%5Cpartial%20z%5E%7Bl%7D%7D)

显然，问题的关键在于求解![\frac{\partial z^{l+1}}{\partial z^{l}}](https://math.jianshu.com/math?formula=%5Cfrac%7B%5Cpartial%20z%5E%7Bl%2B1%7D%7D%7B%5Cpartial%20z%5E%7Bl%7D%7D)，再根据前向传播公式

