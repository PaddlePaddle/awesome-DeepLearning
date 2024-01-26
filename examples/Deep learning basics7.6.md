# 深度学习基础知识

## 1.深度学习发展历史
人工智能是人类最美好的梦想之一。虽然计算机技术已经取得了长足的进步，但是到目前为止，还没有一台电脑能产生“自我”的意识。计算机能够具有人的意识起源于图灵测试（Turing Testing）问题的产生，由“计算机科学之父”及“人工智能之父”英国数学家阿兰·图灵在1950年的一篇著名论文《机器会思考吗？》里提出图灵测试的设想：把一个人和一台计算机分别隔离在两间屋子，然后让屋外的一个提问者对两者进行问答测试。如果提问者无法判断哪边是人，哪边是机器，那就证明计算机已具备人的智能。

但是半个世纪过去了，人工智能的进展，远远没有达到图灵试验的标准。这不仅让多年翘首以待的人们心灰意冷，认为人工智能是忽悠，相关领域是“伪科学”。直到深度学习（Deep Learning）的出现，让人们看到了一丝曙光。

BryantLJ说"学习任一门知识都应该先从其历史开始，把握了历史，也就抓住了现在与未来",在深度学习漫长的发展岁月中，一些取得关键突破的闪光时刻，值得我们这些深度学习爱好者们铭记，如**图1**所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8212741e9a70495ea467a2d2a861baff9ecd964aa67447e8a415b162dd5d01a4" width="900" hegiht="" ></center>
<center><br>图1：深度学习发展历程</br></center>
<br></br>

* **1940年代**：1943年，由神经科学家麦卡洛克(W.S.McCilloch) 和数学家皮兹（W.Pitts）在《数学生物物理学公告》上发表论文《神经活动中内在思想的逻辑演算》（A Logical Calculus of the Ideas Immanent in Nervous Activity）。建立了神经网络和数学模型，称为**MCP模型**。所谓MCP模型，其实是按照生物神经元的结构和工作原理构造出来的一个抽象和简化了的模型，也就诞生了所谓的“模拟大脑”，人工神经网络的大门由此开启。

* **50-60年代**：1958年，计算机科学家罗森布拉特（ Rosenblatt）提出了两层神经元组成的神经网络，称之为“感知器”(Perceptrons)。第一次将MCP用于机器学习（machine learning）分类(classification)。“感知器”算法算法使用MCP模型对输入的多维数据进行二分类，且能够使用梯度下降法从训练样本中自动学习更新权值。1962年,该方法被证明为能够收敛，理论与实践效果引起第一次神经网络的浪潮。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f5ba80aa2de4426d86811fb31d84a1a314d6a791eca44c7f9eefc340b6679063" width="500" hegiht="" ></center>
<center><br>图2： Frank Rosenblatt和感知机的提出</br></center>
<br></br>

* **1969年**：美国数学家及人工智能先驱 Marvin Minsky 在其著作中证明了感知器本质上是一种线性模型（linear model），只能处理线性分类问题，就连最简单的XOR（亦或）问题都无法正确分类，神经网络模型进入了被束之高阁的黑暗时代。

* **1986年**：由神经网络之父 Geoffrey Hinton 在1986年发明了适用于**多层感知器**（MLP）的**BP（Backpropagation）算法**，并采用**Sigmoid**进行非线性映射，有效解决了非线性分类和学习的问题。该方法引起了神经网络的第二次热潮。但是1991年BP算法被指出存在梯度消失问题，也就是说在误差梯度后项传递的过程中，后层梯度以乘性方式叠加到前层，由于Sigmoid函数的饱和特性，后层梯度本来就小，误差梯度传到前层时几乎为0，因此无法对前层进行有效的学习，该问题直接阻碍了深度学习的进一步发展。
此外90年代中期，支持向量机算法诞生（SVM算法）等各种浅层机器学习模型被提出，SVM也是一种有监督的学习模型，应用于模式识别，分类以及回归分析等。支持向量机以统计学为基础，和神经网络有明显的差异，支持向量机等算法的提出再次阻碍了深度学习的发展。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b87b93c0b92e4582a6eb6ac9ec9a21cadeb9f11e9d614db98ef1ad01cf81a394" width="200" hegiht="" ></center>
<center><br>图3：Sigmoid函数</br></center>
<br></br>

* **2010年左右**：2006年，加拿大多伦多大学教授、机器学习领域泰斗、神经网络之父—— Geoffrey Hinton 和他的学生 Ruslan Salakhutdinov 在顶尖学术刊物《科学》上发表了一篇文章，该文章提出了深层网络训练中梯度消失问题的解决方案：无监督预训练对权值进行初始化+有监督训练微调。斯坦福大学、纽约大学、加拿大蒙特利尔大学等成为研究深度学习的重镇，至此开启了深度学习在学术界和工业界的浪潮。
2011年，ReLU激活函数被提出，该激活函数能够有效的抑制梯度消失问题。2011年以来，微软首次将DL应用在语音识别上，取得了重大突破。微软研究院和Google的语音识别研究人员先后采用DNN技术降低语音识别错误率20％~30％，是语音识别领域十多年来最大的突破性进展。2012年，DNN技术在图像识别领域取得惊人的效果，在ImageNet评测上将错误率从26％降低到15％。在这一年，DNN还被应用于制药公司的DrugeActivity预测问题，并获得世界最好成绩。
2012年，Hinton课题组为了证明深度学习的潜力，首次参加ImageNet图像识别比赛，其通过构建的CNN网络AlexNet一举夺得冠军，且碾压第二名（SVM方法）的分类性能。也正是由于该比赛，CNN吸引到了众多研究者的注意。
2013、2014、2015、2016年，通过ImageNet图像识别比赛，DL的网络结构，训练方法，GPU硬件的不断进步，促使其在其他领域也在不断的征服战场。
2016年3月，由谷歌（Google）旗下DeepMind公司开发的AlphaGo(基于深度学习)与围棋世界冠军、职业九段棋手李世石进行围棋人机大战，以4比1的总比分获胜；2016年末2017年初，该程序在中国棋类网站上以“大师”（Master）为注册帐号与中日韩数十位围棋高手进行快棋对决，连续60局无一败绩；2017年5月，在中国乌镇围棋峰会上，它与排名世界第一的世界围棋冠军柯洁对战，以3比0的总比分获胜。围棋界公认阿尔法围棋的棋力已经超过人类职业围棋顶尖水平。


## 2.人工智能、机器学习、深度学习的联系与区别

概括来说，人工智能、机器学习和深度学习覆盖的技术范畴是逐层递减的。人工智能是最宽泛的概念，是整个学科的名称。机器学习是当前比较有效的一种实现人工智能的方式，是人工智能的重要组成部分, 是其的子集，但不是唯一的部分。神经网络是机器学习的一种
分支方法，不过机器学习大家庭下还有其他分支。深度学习是关于构建、训练和使
用神经网络的一种现代方法，深度学习是机器学习算法中最热门的一个分支，近些年取得了显著的进展，并替代了大多数传统机器学习算法。三者的关系如 图4 所示，即：人工智能 > 机器学习

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/80a5e24ace954f9bb643cd08133c778d7060e40e34c7430c9d8b054bc86449b7" width="700" hegiht="" ></center>
<center><br>图4：人工智能、机器学习、深度学习关系图</br></center>
<br></br>

### 机器学习

机器学习是专门研究计算机怎样模拟或实现人类的学习行为，以获取新的知识或技能，重新组织已有的知识结构，使之不断改善自身的性能。卡内基梅隆大学的 Tom Mitchell给出了一个机器学习的定义：一个程序被认为能从经验 $E$ 中学习，解决任务 $T$，达到性能度量值 $P$，当且仅当，
有了经验 $E$ 后，经过 $P$ 评判，程序在处理 $T$ 时的性能有所提升。

机器学习的实现可以分成两步：训练和预测
* **训练**：从具体案例中抽象一般规律，从一定数量的样本（已知模型输入$X$和模型输出$Y$）中，学习输出$Y$与输入$X$的关系。

* **预测**：基于训练得到的$Y$与$X$之间的关系，如出现新的输入$X$，计算出输出$Y$。通常情况下，如果通过模型计算的输出和真实场景的输出一致，则说明模型是有效的。

机器学习的过程分为模型假设、评价函数和优化算法三部分：
* **假设**：世界上的可能关系千千万，漫无目标的试探$Y$~$X$之间的关系显然是十分低效的。因此假设空间先圈定了一个模型能够表达的关系可能。机器还会进一步在假设圈定的圆圈内寻找最优的$Y$~$X$关系，即确定参数$W$。

* **评价**：寻找最优之前，我们需要先定义什么是最优，即评价一个$Y$~$X$关系的好坏的指标。通常衡量该关系是否能很好的拟合现有观测样本，将拟合的误差最小作为优化目标。

* **优化**：设置了评价指标后，就可以在假设圈定的范围内，将使得评价指标最优（损失函数最小/最拟合已有观测样本）的$Y$~$X$关系找出来，这个寻找的方法即为优化算法。

假定$f$是最终期望的"公式"函数，$g$是当前公式所拟合的函数。那么，机器学习的目的就是让假设$g$更加逼近期望目标$f$。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c3fe62484814492f988bac8085497aff15e28652610549d3aee5159264686b90" width="700" hegiht="" ></center>
<center><br>图5：使用数据去计算假设g去逼近目标f</br></center>
<br></br>

### 深度学习

深度学习与传统的机器学习方法相比，两者的理论结构是一致的，即：模型假设、评价函数和优化算法，其根本差别在于假设的复杂度。






## 3.神经元、单层感知机、多层感知机
在生物学中，神经元细胞有兴奋与抑制两种状态。大多数神经元细胞在正常情况下处于抑制状态，一旦某个神经元受到刺激并且电位超过一定的阈值后，这个神经元细胞就被激活，处于兴奋状态，并向其他神经元传递信息。基于神经元细胞的结构特性与传递信息方式，神经科学家 Warren McCulloch 和逻辑学家 Walter Pitts 合作提出了“McCulloch–Pitts (MCP) neuron”模型。在人工神经网络中，MCP模型成为人工神经网络中的最基本结构。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/cfa3b0d767f943929b3b2115738dccd0a29f4f066b8f4ef7a18b70ded0bdb153" width="700" hegiht="" ></center>
<center><br>图6：MCP模型示意图</br></center>
<br></br>

* **神经元：** 神经网络中每个节点称为神经元，由两部分组成：
  - 加权和：将所有输入加权求和。
  - 非线性变换（激活函数）：加权和的结果经过一个非线性函数变换，让神经元计算具备非线性的能力。
  
* **单层感知机：**
1957年 Frank Rosenblatt 提出了一种简单的人工神经网络，被称之为感知机。早期的感知机结构和 MCP 模型相似，由一个输入层和一个输出层构成，因此也被称为“单层感知机”。感知机的输入层负责接收实数值的输入向量，输出层则为1或-1两个值。单层感知机可作为一种二分类线性分类模型。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/86185c477655439c9ce36a4afa99ab9a3dfb7763fe064f2f8d11296e395030fb" width="300" hegiht="" ></center>
<center><br>图7：单层感知机</br></center>
<br></br>

感知机的数学模型为：
$$o =sign(w_ix_i+b)$$

其中， $w$ = ( $w_1$,$w_2$,...,$w_n$ $)$   是权重向量,$b$是偏置,$x_i$=( $x_i^1$,$x_i^2$,...$x_i^n$ ) 是样本点。$sign$表示符号函数，对于 $t$>0,有$sign$($t$)=1,反之$sign(t)=-1$。

* **多层感知机：** 由于无法模拟诸如异或以及其他复杂函数的功能，使得单层感知机的应用较为单一。一个简单的想法是，如果能在感知机模型中增加若干隐藏层，增强神经网络的非线性表达能力，就会让神经网络具有更强拟合能力。因此，由多个隐藏层构成的多层感知机被提出。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/801d3e7325ce4e4a97ae4f290759457010f4a98e8b864f4196cb1e56afc6adcf" width="300" hegiht="" ></center>
<center><br>图8：多层感知机</br></center>
<br></br>


这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元。输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算；因此，这个多层感知机中的层数为2。$\mathbf{X} \in \mathbb{R}^{n \times d}$表示$n$个样本，每个样本由$d$个特征 。对于具有$h$个隐藏单元的单隐藏层多层感知机，用$\mathbf{H} \in \mathbb{R}^{n \times h}$表示隐藏层的输出，隐藏层权重 $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$和隐藏层偏置$\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$以及输出层权重$\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$和输出层偏置$\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$。
为了发挥多层结构的潜力，还需要在仿射变换之后对每个隐藏单元应用非线性的激活函数（activation function）$\sigma$。
最终，多层感知机模型为：


$$\mathbf{H}  = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$$

$$\mathbf{O}  = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$



## 5.什么是前向传播

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8112a66c6251401ca827aec866b3daa8cc42cf38f638425b8613edd9b3a41c43" width="500" hegiht="" ></center>
<center><br>图9：前向传播权重示意图</br></center>
<br></br>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e8c8cd56a83849a88b4b85fbcf2421d8b2e16b7bc50a4bf0ae8a943c07763908" width="200" hegiht="" ></center>
<center><br>图10：前向偏置传播示意图</br></center>
<br></br>



记$w_{jk}^{l}$为第$l−1$层第$k$个神经元到第$l$层第$j$个神经元的权重，$b_j^l$为第$l$层第$j$个神经元的偏置，$a_j^l$为第$l$层第$j$个神经元的激活值（激活函数的输出）。$a_j^l$的值取决于上一层神经元的激活：

$$a_j^l=\sigma{(\sum_k{w_{jk}^l a_k^{l-1}}+b_j^l)} $$

将上式重写为矩阵形式：
$$a^l=\sigma{(w^l a^{l-1} +b^l)}$$

为了方便表示，记 $z^l=w^l a^{l-1} +b^l$为每一层的权重输入， 上式则变为 $a^l=\sigma{(z^l)}$。
　　


利用上式一层层计算网络的激活值，最终能够根据输入 $X$得到相应的输出$\hat Y$。


## 6.什么是反向传播
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ab8337d928e24300a0cc71a914c1833d2cf58568bc4f486a9d92c68973f9bc67" width="200" hegiht="" ></center>
<center><br>图10：代价函数</br></center>
<br></br>



![](https://ai-studio-static-online.cdn.bcebos.com/ab8337d928e24300a0cc71a914c1833d2cf58568bc4f486a9d92c68973f9bc67)

权重$w$和偏置$b$的改变如何影响代价函数$C$是理解反向传播的关键。$\delta_j^l$表示第$l$层第$j$个单元的误差,则:
$$\delta_j^l=\frac{\partial{C}}{\partial z_j^l}$$

### 四个基本方程：
**输出层的误差方程**
$$\delta_j^L=\frac{\partial C}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\frac{\partial a_j^L}{\partial z_j^L}=\frac{\partial C}{\partial a_j^L}\sigma'(z_j^L)$$
如果代价函数$C=\frac{1}{2}  \sum_j{(y_j - a_j^L)^2}$，则$\frac{\partial C}{\partial a_j^L}=a_j^L-y_j$,同理，对激活函数$\sigma(z)$求$z_j^L$的偏导即可求得$\sigma'(z_j^L)$。则上式可以写为：
$$\delta^L=\nabla_aC \odot \sigma'(z^L)$$

$\odot$为Hadamard积，即矩阵的点积。

**误差传递方程**
$$delta^l=((w^{l+1})^T\delta^{l+1})\odot \sigma'(z^l)$$
证明过程如下：
$$\delta_j^l=\frac{\partial C}{\partial z_j^l} = \sum_k \frac{\partial C}{\partial z_k^{l+1}} \frac{\partial z_k^{l+1}}{\partial z_j^l}= \sum_k \delta_k^{l+1} \frac{\partial z_k^{l+1}}{\partial z_j^l}$$
因为$z_k^{l+1}=\sum_j{w_{kj}^{l+1}a_j^l+b_k^{l+1}}=\sum_j{w_{kj}^{l+1}\sigma{(z_j^l)}+b_k^{l+1}}$，所以$\frac{\partial z_k^{l+1}}{\partial z_j^l}=w_{kj}^{l+1}\sigma'(z_j^l)$,因此可以得到:
$$\delta_j^l=\sum_k w_{kj}^{l+1} \delta_k^{l+1} \sigma'(z_j^l)$$

**代价函数对偏置的改变率**
$$\frac{\partial C}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^l}{\partial b_j^l}=\frac{\partial C}{\partial z_j^l}=\delta_j^l $$
这里因为$z_j^l=\sum_k{w_{jk}^l a_k^{l-1}}+b_j^l$,所以$\frac{\partial z_j^L}{\partial b_j^L}=1$
**代价函数对偏置的改变率**
$$\frac{\partial C}{\partial w_{jk}^l}=\frac{\partial C}{\partial z_j^l}\frac{\partial z_j^L}{\partial w_{jk}^l}=\frac{\partial C}{\partial z_j^l}a_k^{l-1}=a_k^{l-1}\delta_j^l$$

可以简写为:
$$\frac{\partial C}{\partial w}=a_{in}\delta_{out}$$
当上一层激活输出接近0的时候，无论返回的误差有多大，$\frac{\partial C}{\partial w}$的改变都很小，这也就解释了为什么神经元饱和不利于训练。

从上面的推导我们不难发现，当输入神经元没有被激活，或者输出神经元处于饱和状态，权重和偏置会学习的非常慢，这不是我们想要的效果。这也说明了为什么我们平时总是说激活函数的选择非常重要。









```python

```
