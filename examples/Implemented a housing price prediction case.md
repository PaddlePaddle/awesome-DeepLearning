**1.深度学习的基础知识**
**①深度学习发展历史**
![](https://ai-studio-static-online.cdn.bcebos.com/2f48157518b443488f2e1404ed96cb0832fd0d86424f4374ae8bbf8274b3ff49)
   由图可以明显看出DL在从06年崛起之前经历了两个低谷，这两个低谷也将神经网络的发展分为了三个不同的阶段，下面就分别讲述这三个阶段。
**第一代神经网络（1958~1969）**

最早的神经网络的思想起源于1943年的MCP人工神经元模型，当时是希望能够用计算机来模拟人的神经元反应的过程，该模型将神经元简化为了三个过程：输入信号线性加权，求和，非线性激活（阈值法）。如下图所示：![](https://ai-studio-static-online.cdn.bcebos.com/65170df3624d4f2d83e947913c0ba9ad5f90eda75cf6410db3596c2ef966d463)
第一次将MCP用于机器学习（分类）的当属1958年Rosenblatt发明的感知器（perceptron）算法。该算法使用MCP模型对输入的多维数据进行二分类，且能够使用梯度下降法从训练样本中自动学习更新权值。1962年，该方法被证明为能够收敛，理论与实践效果引起第一次神经网络的浪潮。

然而学科发展的历史不总是一帆风顺的。

1969年，美国数学家及人工智能先驱Minsky在其著作中证明了感知器本质上是一种线性模型，只能处理线性分类问题，就连最简单的XOR（亦或）问题都无法正确分类。这等于直接宣判了感知器的死刑，神经网络的研究也陷入了近20年的停滞。

**第二代神经网络（1986~1998）**

第一次打破非线性诅咒的当属现代DL大牛Hinton，其在1986年发明了适用于多层感知器（MLP）的BP算法，并采用Sigmoid进行非线性映射，有效解决了非线性分类和学习的问题。该方法引起了神经网络的第二次热潮。

1989年，Robert Hecht-Nielsen证明了MLP的万能逼近定理，即对于任何闭区间内的一个连续函数f，都可以用含有一个隐含层的BP网络来逼近该定理的发现极大的鼓舞了神经网络的研究人员。

也是在1989年，LeCun发明了卷积神经网络-LeNet，并将其用于数字识别，且取得了较好的成绩，不过当时并没有引起足够的注意。

值得强调的是在1989年以后由于没有特别突出的方法被提出，且NN一直缺少相应的严格的数学理论支持，神经网络的热潮渐渐冷淡下去。冰点来自于1991年，BP算法被指出存在梯度消失问题，即在误差梯度后向传递的过程中，后层梯度以乘性方式叠加到前层，由于Sigmoid函数的饱和特性，后层梯度本来就小，误差梯度传到前层时几乎为0，因此无法对前层进行有效的学习，该发现对此时的NN发展雪上加霜。

1997年，LSTM模型被发明，尽管该模型在序列建模上的特性非常突出，但由于正处于NN的下坡期，也没有引起足够的重视。

**统计学习方法的春天（1986~2006）**

1986年，决策树方法被提出，很快ID3，ID4，CART等改进的决策树方法相继出现，到目前仍然是非常常用的一种机器学习方法。该方法也是符号学习方法的代表。 

1995年，线性SVM被统计学家Vapnik提出。该方法的特点有两个：由非常完美的数学理论推导而来（统计学与凸优化等），符合人的直观感受（最大间隔）。不过，最重要的还是该方法在线性分类的问题上取得了当时最好的成绩。 

1997年，AdaBoost被提出，该方法是PAC（Probably Approximately Correct）理论在机器学习实践上的代表，也催生了集成方法这一类。该方法通过一系列的弱分类器集成，达到强分类器的效果。 

2000年，KernelSVM被提出，核化的SVM通过一种巧妙的方式将原空间线性不可分的问题，通过Kernel映射成高维空间的线性可分问题，成功解决了非线性分类的问题，且分类效果非常好。至此也更加终结了NN时代。 

2001年，随机森林被提出，这是集成方法的另一代表，该方法的理论扎实，比AdaBoost更好的抑制过拟合问题，实际效果也非常不错。 

2001年，一种新的统一框架-图模型被提出，该方法试图统一机器学习混乱的方法，如朴素贝叶斯，SVM，隐马尔可夫模型等，为各种学习方法提供一个统一的描述框架。

**第三代神经网络-DL（2006-至今）**

该阶段又分为两个时期：快速发展期（2006~2012）与爆发期（2012~至今）

*快速发展期（2006~2012）*

2006年，DL元年。是年，Hinton提出了深层网络训练中梯度消失问题的解决方案：无监督预训练对权值进行初始化+有监督训练微调。其主要思想是先通过自学习的方法学习到训练数据的结构（自动编码器），然后在该结构上进行有监督训练微调。但是由于没有特别有效的实验验证，该论文并没有引起重视。

2011年，ReLU激活函数被提出，该激活函数能够有效的抑制梯度消失问题。

2011年，微软首次将DL应用在语音识别上，取得了重大突破。

*爆发期（2012~至今）*

2012年，Hinton课题组为了证明深度学习的潜力，首次参加ImageNet图像识别比赛，其通过构建的CNN网络AlexNet一举夺得冠军，且碾压第二名（SVM方法）的分类性能。也正是由于该比赛，CNN吸引到了众多研究者的注意。 

AlexNet的创新点： 

（1）首次采用ReLU激活函数，极大增大收敛速度且从根本上解决了梯度消失问题；（2）由于ReLU方法可以很好抑制梯度消失问题，AlexNet抛弃了“预训练+微调”的方法，完全采用有监督训练。也正因为如此，DL的主流学习方法也因此变为了纯粹的有监督学习；（3）扩展了LeNet5结构，添加Dropout层减小过拟合，LRN层增强泛化能力/减小过拟合；（4）首次采用GPU对计算进行加速；

2013,2014,2015年，通过ImageNet图像识别比赛，DL的网络结构，训练方法，GPU硬件的不断进步，促使其在其他领域也在不断的征服战场

2015年，Hinton，LeCun，Bengio论证了局部极值问题对于DL的影响，结果是Loss的局部极值问题对于深层网络来说影响可以忽略。该论断也消除了笼罩在神经网络上的局部极值问题的阴霾。具体原因是深层网络虽然局部极值非常多，但是通过DL的BatchGradientDescent优化方法很难陷进去，而且就算陷进去，其局部极小值点与全局极小值点也是非常接近，但是浅层网络却不然，其拥有较少的局部极小值点，但是却很容易陷进去，且这些局部极小值点与全局极小值点相差较大。论述原文其实没有证明，只是简单叙述，严密论证是猜的。。。

2015，DeepResidualNet发明。分层预训练，ReLU和BatchNormalization都是为了解决深度神经网络优化时的梯度消失或者爆炸问题。但是在对更深层的神经网络进行优化时，又出现了新的Degradation问题，即”通常来说，如果在VGG16后面加上若干个单位映射，网络的输出特性将和VGG16一样，这说明更深次的网络其潜在的分类性能只可能>=VGG16的性能，不可能变坏，然而实际效果却是只是简单的加深VGG16的话，分类性能会下降（不考虑模型过拟合问题）“Residual网络认为这说明DL网络在学习单位映射方面有困难，因此设计了一个对于单位映射（或接近单位映射）有较强学习能力的DL网络，极大的增强了DL网络的表达能力。此方法能够轻松的训练高达150层的网络。

**②人工智能、机器学习、深度学习有什么区别和联系？**
**人工智能：从概念提出到走向繁荣**
   1956年，几个计算机科学家相聚在达特茅斯会议，提出了“人工智能”的概念，梦想着用当时刚刚出现的计算机来构造复杂的、拥有与人类智慧同样本质特性的机器。其后，人工智能就一直萦绕于人们的脑海之中，并在科研实验室中慢慢孵化。之后的几十年，人工智能一直在两极反转，或被称作人类文明耀眼未来的预言，或被当成技术疯子的狂想扔到垃圾堆里。直到2012年之前，这两种声音还在同时存在。
   2012年以后，得益于数据量的上涨、运算力的提升和机器学习新算法（深度学习）的出现，人工智能开始大爆发。据领英近日发布的《全球AI领域人才报告》显示，截至2017年一季度，基于领英平台的全球AI（人工智能）领域技术人才数量超过190万，仅国内人工智能人才缺口达到500多万。
   人工智能的研究领域也在不断扩大，图二展示了人工智能研究的各个分支，包括专家系统、机器学习、进化计算、模糊逻辑、计算机视觉、自然语言处理、推荐系统等。
   但目前的科研工作都集中在弱人工智能这部分，并很有希望在近期取得重大突破，电影里的人工智能多半都是在描绘强人工智能，而这部分在目前的现实世界里难以真正实现（通常将人工智能分为弱人工智能和强人工智能，前者让机器具备观察和感知的能力，可以做到一定程度的理解和推理，而强人工智能让机器获得自适应能力，解决一些之前没有遇到过的问题）。弱人工智能有希望取得突破，是如何实现的，“智能”又从何而来呢？这主要归功于一种实现人工智能的方法——机器学习。

**机器学习：一种实现人工智能的方法**
   机器学习最基本的做法，是使用算法来解析数据、从中学习，然后对真实世界中的事件做出决策和预测。与传统的为解决特定任务、硬编码的软件程序不同，机器学习是用大量的数据来“训练”，通过各种算法从数据中学习如何完成任务。
   举个简单的例子，当我们浏览网上商城时，经常会出现商品推荐的信息。这是商城根据你往期的购物记录和冗长的收藏清单，识别出这其中哪些是你真正感兴趣，并且愿意购买的产品。这样的决策模型，可以帮助商城为客户提供建议并鼓励产品消费。
   机器学习直接来源于早期的人工智能领域，传统的算法包括决策树、聚类、贝叶斯分类、支持向量机、EM、Adaboost等等。从学习方法上来分，机器学习算法可以分为监督学习（如分类问题）、无监督学习（如聚类问题）、半监督学习、集成学习、深度学习和强化学习。
   传统的机器学习算法在指纹识别、基于Haar的人脸检测、基于HoG特征的物体检测等领域的应用基本达到了商业化的要求或者特定场景的商业化水平，但每前进一步都异常艰难，直到深度学习算法的出现。
   
**深度学习：一种实现机器学习的技术**
   深度学习本来并不是一种独立的学习方法，其本身也会用到有监督和无监督的学习方法来训练深度神经网络。但由于近几年该领域发展迅猛，一些特有的学习手段相继被提出（如残差网络），因此越来越多的人将其单独看作一种学习的方法。
   最初的深度学习是利用深度神经网络来解决特征表达的一种学习过程。深度神经网络本身并不是一个全新的概念，可大致理解为包含多个隐含层的神经网络结构。为了提高深层神经网络的训练效果，人们对神经元的连接方法和激活函数等方面做出相应的调整。其实有不少想法早年间也曾有过，但由于当时训练数据量不足、计算能力落后，因此最终的效果不尽如人意。
   深度学习摧枯拉朽般地实现了各种任务，使得似乎所有的机器辅助功能都变为可能。无人驾驶汽车，预防性医疗保健，甚至是更好的电影推荐，都近在眼前，或者即将实现。

**三者的区别和联系**
   机器学习是一种实现人工智能的方法，深度学习是一种实现机器学习的技术。我们就用最简单的方法——同心圆，可视化地展现出它们三者的关系。
   目前，业界有一种错误的较为普遍的意识，即“深度学习最终可能会淘汰掉其他所有机器学习算法”。这种意识的产生主要是因为，当下深度学习在计算机视觉、自然语言处理领域的应用远超过传统的机器学习方法，并且媒体对深度学习进行了大肆夸大的报道。
   深度学习，作为目前最热的机器学习方法，但并不意味着是机器学习的终点。起码目前存在以下问题：
   1. 深度学习模型需要大量的训练数据，才能展现出神奇的效果，但现实生活中往往会遇到小样本问题，此时深度学习方法无法入手，传统的机器学习方法就可以处理；
   2. 有些领域，采用传统的简单的机器学习方法，可以很好地解决了，没必要非得用复杂的深度学习方法；
   3. 深度学习的思想，来源于人脑的启发，但绝不是人脑的模拟，举个例子，给一个三四岁的小孩看一辆自行车之后，再见到哪怕外观完全不同的自行车，小孩也十有八九能做出那是一辆自行车的判断，也就是说，人类的学习过程往往不需要大规模的训练数据，而现在的深度学习方法显然不是对人脑的模拟。

**③神经元、单层感知机、多层感知机**
**神经元**
一个神经元通常具有多个树突，主要用来接受传入信息；而轴突只有一条，轴突尾端有许多轴突末梢可以给其他多个神经元传递信息。轴突末梢跟其他神经元的树突产生连接，从而传递信号。这个连接的位置在生物学上叫做“突触”。突触之间的交流通过神经递质实现。
下面对上面的这个模型进行抽象处理。首先考虑到神经元结构有多个树突，一个轴突可将其抽象为下图的黑箱结构：![](https://ai-studio-static-online.cdn.bcebos.com/04ef3c87c76943068678b884ce3299013703cefc5f4e441884472f3af4d5a383)
但是黑箱结构有诸多不便，首先是不知道黑箱中的函数结构就不能为我们所用，其次是输入输出与黑箱的关系也无法量化。因此考虑将上述结构简化，首先把树突到细胞核的阶段简化为线性加权的过程（当然了，该过程也有可能是非线性的，但是我们可以把其非线性过程施加到后面的非线性函数以及多层网络结构中），其次把突触之间的信号传递简化为对求和结果的非线性变换，那么上述模型就变得清晰了：
![](https://ai-studio-static-online.cdn.bcebos.com/1fb1cff3a5ed496ca381202fe35e09acad168822a11840b0956800994fb7cf95)

**单层感知机**
面我们介绍的神经元的基本模型实际就是一个感知机的模型，该词最早出现于1958年，计算科学家Rosenblatt提出的由两层神经元组成的神经网络。
![](https://ai-studio-static-online.cdn.bcebos.com/bb14be1dbf294204a9975b5b66d034652cbea2638ea84e7ca0ca95352622d4fa)对前面的模型进一步符号化，如下图所示：

可以看到，感知机的基本模型包括：
输入：x1,x2...xn,实际可能回比这更多,此处添加了一个偏置1，是为了平衡线性加权函数总是过零点的问题;
权值：对应于每个输入都有一个加权的权值w1,w2...wN;
激活函数：激活函数f对应于一个非线性函数，其选择有很多，本文后面会详细介绍;
输出y：由激活激活函数进行处理后的结果，往往是区分度较大的非连续值用于分类;

激活函数
   首先我们来了解一下激活函数有什么意义，我们前面提到，我们把权重加权的过程看作线性加权，此时该系统为一线性系统，能够解决的问题有限，其非线性部分在激活函数中体现。使用激活函数能够使神经网络更加适应生活实际中的非线性问题。
   
**多层感知机**
 多层感知机（MLP，Multilayer Perceptron）也叫人工神经网络（ANN，Artificial Neural Network），除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构，如下图：![](https://ai-studio-static-online.cdn.bcebos.com/412dc6d7699541d1aa7dbca21a31bca0352008cdcf3345e5b8e78306e4ec3f37)
   从上图可以看到，多层感知机层与层之间是全连接的。多层感知机最底层是输入层，中间是隐藏层，最后是输出层。 
   隐藏层的神经元怎么得来？首先它与输入层是全连接的，假设输入层用向量X表示，则隐藏层的输出就是 f (W1X+b1)，W1是权重（也叫连接系数），b1是偏置，函数f 可以是常用的sigmoid函数或者tanh函数：

**④什么是前向传播**
![](https://ai-studio-static-online.cdn.bcebos.com/9d48986be2d44ea5aef3a312a97f63c51847b839ad73430eb11e21ee0aab5bc9)
   如图所示，这里讲得已经很清楚了，前向传播的思想比较简单。
举个例子，假设上一层结点i,j,k,…等一些结点与本层的结点w有连接，那么结点w的值怎么算呢？就是通过上一层的i,j,k等结点以及对应的连接权值进行加权和运算，最终结果再加上一个偏置项（图中为了简单省略了），最后在通过一个非线性函数（即激活函数），如ReLu，sigmoid等函数，最后得到的结果就是本层结点w的输出。
最终不断的通过这种方法一层层的运算，得到输出层结果。

**⑤什么是反向传播**
BackPropagation算法是多层神经网络的训练中举足轻重的算法。简单的理解，它的确就是复合函数的链式法则，但其在实际运算中的意义比链式法则要大的多。要回答题主这个问题“如何直观的解释back propagation算法？” 需要先直观理解多层神经网络的训练。

机器学习可以看做是数理统计的一个应用，在数理统计中一个常见的任务就是拟合，也就是给定一些样本点，用合适的曲线揭示这些样本点随着自变量的变化关系.

深度学习同样也是为了这个目的，只不过此时，样本点不再限定为(x, y)点对，而可以是由向量、矩阵等等组成的广义点对(X,Y)。而此时，(X,Y)之间的关系也变得十分复杂，不太可能用一个简单函数表示。然而，人们发现可以用多层神经网络来表示这样的关系，而多层神经网络的本质就是一个多层复合的函数。借用网上找到的一幅图[1]，来直观描绘一下这种复合关系。![](https://ai-studio-static-online.cdn.bcebos.com/43390b4a59734ae2b1881a9a195bd98d74758588142e4ffaa9c1cb9a57535976)
其对应的表达式如下：![](https://ai-studio-static-online.cdn.bcebos.com/33bb98133d58489c86508eced99d10f10dab9bfaffb24585bce1effaad4b739c)

**房价预测的python+numpy实现**
**1. 解压数据集，并且查看打印数据集内容**
房价数据集只有两个标签，分别是房屋的面积和房价，为了便于处理，我们对数据进行归一化处理操作。 归一化的方式有多种，此次我们采用max-min归一化操作。

基本上所有的数据在拿到后都必须进行归一化，至少有以下3条原因：

1.过大或过小的数值范围会导致计算时的浮点上溢或下溢。

2.不同的数值范围会导致不同属性对模型的重要性不同（至少在训练的初始阶段如此），而这个隐含的假设常常是不合理的。这会对优化的过程造成困难，使训练时间大大加长。

3.很多的机器学习技巧/模型（例如L1，L2正则项，向量空间模型-Vector Space Model）都基于这样的假设：所有的属性取值都差不多是以0为均值且取值范围相近的。

**数据集分割**

将原始数据处理为可用数据后，为了评估模型的好坏，我们将数据分成两份：训练集和测试集。

训练集数据用于调整模型的参数，即进行模型的训练，模型在这份数据集上的误差被称为训练误差； 测试集数据被用来测试，模型在这份数据集上的误差被称为测试误差。 我们训练模型的目的是为了通过从训练数据中找到规律来预测未知的新数据，所以测试误差是更能反映模型表现的指标。分割数据的比例要考虑到两个因素：更多的训练数据会降低参数估计的方差，从而得到更可信的模型；而更多的测试数据会降低测试误差的方差，从而得到更可信的测试误差。我们这个例子中设置的分割比例为8:2


```python

```


```python
import numpy as np
import pandas as pd
data_path='./房价预测/data/data.txt'
colnames = ['房屋面积']+['房价']
print_data = pd.read_csv(data_path, names = colnames)
data=np.loadtxt(data_path,delimiter = ',')
print_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>房屋面积</th>
      <th>房价</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>98.87</td>
      <td>599.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>68.74</td>
      <td>450.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>89.24</td>
      <td>440.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>129.19</td>
      <td>780.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61.64</td>
      <td>450.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#axis=0,表示按列计算
#data.shape[0]表示data中一共有多少列
maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
print("the raw area :",data[:,0].max(axis = 0))
print("按列最大值，最小值，平均值：",maximums,minimums,avgs)
data=(data-avgs)/(maximums-minimums)
print('normalizatin:',data.max(axis=0))
```

    the raw area : 199.96
    按列最大值，最小值，平均值： [ 199.96 2000.  ] [ 40.09 202.  ] [ 94.64454023 608.25057471]
    normalizatin: [0.65875686 0.77405419]



```python
### 数据集的分割，ratio为分割比例
ratio = 0.8
offset = int(data.shape[0]*ratio)
train_data = data[:offset].copy()
test_data = data[offset:].copy()
print(len(data))
print(len(train_data))
```

    870
    696



```python
#对上述函数进行封装
def data_load(data_path,ratio=0.8):
    data=np.loadtxt(data_path,delimiter = ',',dtype='float32')
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
    #归一化
    data=(data-avgs)/(maximums-minimums)
    #数据集分割
    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    train_data = data[:offset].copy()
    test_data = data[offset:].copy()
    train_feature,train_label=train_data[:,:-1],train_data[:,-1]
    test_feature,test_label=test_data[:,:-1],test_data[:,-1]
    return train_feature,train_label,test_feature,test_label
```

**2.模型设计**
假设房价和各影响因素之间能够用线性关系来描述：![](https://ai-studio-static-online.cdn.bcebos.com/7cc5a55596704e408e6a7cd917e6dac2d0529f02403b45b59832f32e78d90608)
我们可以直接给定一组初始化的系数W，然后进行模型的拟合。 损失函数我们采用MSE损失函数，方程式如下所示。为了计算方便，我们更改系数为1/2n![](https://ai-studio-static-online.cdn.bcebos.com/3f167047b6f544679f0278e597b0969f0228f649395840b69c2db28918e5a754)
本次实验所采用的线性模型的结构如下所示。由于numpy实现两层的过于复杂，时间关系，仅通过一层来实现作为说明


```python
# 自定义网络框架
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
    # 前向计算过程    
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    # MSE函数实现
    def loss(self, z, y):
            error = z - y
            cost = error * error
            cost = np.mean(cost)
            return cost
x,y,_,_=data_load(data_path)

net=Network(1)
z=net.forward(x[:5])
loss=net.loss(z,y[:5])
print(f' predict：{z[:,0]}\n',
f"Ture label:{y[:5]}\n",
"loss:",loss)
```

     predict：[ 0.04662517 -0.2858381  -0.05963511  0.38118489 -0.36418156]
     Ture label:[-0.00514491 -0.08801477 -0.0935765   0.0955225  -0.08801477]
     loss: 0.0757117481668684


**3.训练过程**
我们采用梯度下降法来对我们的算法参数进行优化求解![](https://ai-studio-static-online.cdn.bcebos.com/64da0182091342f1a13e5d076d78510f50f60ae8e7ee4afeab696a7058e3040a)
对上图我们还可以进行一系列的优化，例如我们可以在X所有项加上一个指定参数1，使之变成X=[1,x1,x2,x3,...xn]T,那么我们对应的可以更改W为[b,w1,w2,w,3,...,wn]T,那么我们的自变量和参数都可以变为nx1的变量。所以线性方程变为y=W_T*X，其中W_T是W的转置，X是表示输入样本。那么我们对于W的修改就可以通过矩阵操作统一实现。



```python
# 对x进行修正
def modify_x(x):
    x=np.column_stack((np.ones((x.shape[0],1)),x))
    return x
# 自定义网络框架的修改版
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
    # 前向计算过程    
    def forward(self, x):
        self.w[0]=0.
        z = np.dot(x, self.w)
        return z
    # MSE函数实现
    def loss(self, z, y):
            error = z - y
            cost = error * error
            cost = np.mean(cost)
            return cost
#计算当前的梯度值
def gradient(x,y,z):
    gradient_w=np.dot(x.T,(z-y))/x.shape[0]
    return gradient_w[:,0]
# x=modify_x(x)
# y=y.reshape(y.shape[0],1)
# net=Network(x.shape[1])
# z=net.forward(x)
# print(x.shape)
# print(y.shape)
# print(z.shape)
# gradient_w=gradient(x,y,z)
# print(gradient_w)
```


```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights+1, 1)
    # 前向计算过程    
    def forward(self, x):
        self.w[0]=0.
        z = np.dot(x, self.w)
        return z
    # MSE函数实现
    def loss(self, z, y):
            error = z - y
            cost = error * error
            cost = np.mean(cost)
            return cost
    def modify_x(self,x):
        x=np.column_stack((np.ones((x.shape[0],1)),x))
        return x
    def gradient(self,x,y):
        y=y.reshape(y.shape[0],1)
        z=self.forward(x)
        gradient_w=np.dot(x.T,(z-y))/x.shape[0]
        return gradient_w
    def step(self,gradient_w,lr):
        self.w=self.w-lr*gradient_w
    def train(self,x,y,iteration=1000,lr=0.5):
        points=[]
        losses=[]
        for i in range(iteration):
            points.append(self.w)
            z=self.forward(x)
            loss=self.loss(z,y)
            gradient_w=self.gradient(x,y)
            losses.append(loss)
            net.step(gradient_w,lr)
            if (i)%50==0:
                print(f'iter: {i}, point: {self.w[:,0]}, loss: {loss}')
            # print(f'iter: {i}, point: {self.w.shape}, loss: {loss}')
            
        return points, losses
```

**3.模型的训练和可视化**


```python
net=Network(x.shape[1])
x_=net.modify_x(x)
y=y.reshape(y.shape[0],1)
iteration=2000
points, losses=net.train(x_,y,iteration)
```

    iter: 0, point: [0.0005321  0.40503304], loss: 0.009834716345136362
    iter: 50, point: [0.00086141 0.55375317], loss: 0.007840170584324837
    iter: 100, point: [0.0009755  0.60527637], loss: 0.0076007783574149265
    iter: 150, point: [0.00101502 0.62312627], loss: 0.007572045680817139
    iter: 200, point: [0.00102872 0.62931026], loss: 0.007568597086369735
    iter: 250, point: [0.00103346 0.63145267], loss: 0.007568183174229592
    iter: 300, point: [0.0010351  0.63219489], loss: 0.007568133495074106
    iter: 350, point: [0.00103567 0.63245203], loss: 0.0075681275324113815
    iter: 400, point: [0.00103587 0.63254112], loss: 0.007568126816752138
    iter: 450, point: [0.00103594 0.63257198], loss: 0.00756812673085626
    iter: 500, point: [0.00103596 0.63258267], loss: 0.007568126720546742
    iter: 550, point: [0.00103597 0.63258637], loss: 0.007568126719309357
    iter: 600, point: [0.00103597 0.63258766], loss: 0.007568126719160844
    iter: 650, point: [0.00103597 0.6325881 ], loss: 0.007568126719143018
    iter: 700, point: [0.00103597 0.63258826], loss: 0.007568126719140878
    iter: 750, point: [0.00103597 0.63258831], loss: 0.007568126719140622
    iter: 800, point: [0.00103597 0.63258833], loss: 0.0075681267191405905
    iter: 850, point: [0.00103597 0.63258833], loss: 0.007568126719140587
    iter: 900, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 950, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1000, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1050, point: [0.00103597 0.63258834], loss: 0.007568126719140585
    iter: 1100, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1150, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1200, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1250, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1300, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1350, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1400, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1450, point: [0.00103597 0.63258834], loss: 0.007568126719140587
    iter: 1500, point: [0.00103597 0.63258834], loss: 0.007568126719140585
    iter: 1550, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1600, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1650, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1700, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1750, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1800, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1850, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1900, point: [0.00103597 0.63258834], loss: 0.007568126719140584
    iter: 1950, point: [0.00103597 0.63258834], loss: 0.007568126719140584



```python
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import mpl  
mpl.rcParams['font.sans-serif']=['SimHei'] # 指定默认字体 
```


```python
plt.figure()
plt.plot(losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>



    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/font_manager.py:1331: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))



![png](output_15_2.png)



```python
plt.figure()
z=net.forward(x_)
x__=x*(maximums[0]-minimums[0])+avgs[0]
y_=y*(maximums[1]-minimums[1])+avgs[1]
z_=z*(maximums[1]-minimums[1])+avgs[1]
plt.scatter(x__,y_)
plt.plot(x__,z_,c='r')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
```


![png](output_16_0.png)


**房价预测的Paddle框架实现**
原理部分与上相同，但此次采用的是双层的神经网络模型，隐层数量为5。


```python
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random
```


```python
class Network(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self,num):
        # 初始化父类中的一些参数
        super(Network, self).__init__()
        
        # 定义一层全连接层，输入维度是num，输出维度是1
        self.fc1 = Linear(in_features=num, out_features=5)
        self.a=paddle.nn.Sigmoid()
        self.fc2 = Linear(5,out_features=1)
    
    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc1(inputs)
        x=self.a(x)
        x=self.fc2(x)
        return x
```


```python
#对上述函数进行封装
def data_load_new(data_path,ratio=0.8):
    data=np.loadtxt(data_path,delimiter = ',',dtype='float32')
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]
    #归一化
    data=(data-avgs)/(maximums-minimums)
    #数据集分割
    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    train_data = data[:offset].copy()
    test_data = data[offset:].copy()
    return train_data,test_data
```


```python
# 声明定义好的线性回归模型
model = Network(x.shape[1])
# 开启模型训练模式
model.train()
# 加载数据
training_data, test_data = data_load_new(data_path)
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
print(train_data.shape,test_data.shape)
```

    (696, 2) (174, 2)



```python
EPOCH_NUM = 10   # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含16条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1]) # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        
        # print(features)

        # 前向计算
        predicts = model(features)
        
        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        # print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        
        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()
```

    epoch: 0, iter: 0, loss is: [0.09917273]
    epoch: 0, iter: 20, loss is: [0.0737596]
    epoch: 0, iter: 40, loss is: [0.01595156]
    epoch: 0, iter: 60, loss is: [0.02178749]
    epoch: 1, iter: 0, loss is: [0.02515522]
    epoch: 1, iter: 20, loss is: [0.02730075]
    epoch: 1, iter: 40, loss is: [0.00904213]
    epoch: 1, iter: 60, loss is: [0.01497635]
    epoch: 2, iter: 0, loss is: [0.02712169]
    epoch: 2, iter: 20, loss is: [0.0274705]
    epoch: 2, iter: 40, loss is: [0.04888221]
    epoch: 2, iter: 60, loss is: [0.01234901]
    epoch: 3, iter: 0, loss is: [0.00879118]
    epoch: 3, iter: 20, loss is: [0.02383676]
    epoch: 3, iter: 40, loss is: [0.03809622]
    epoch: 3, iter: 60, loss is: [0.00925345]
    epoch: 4, iter: 0, loss is: [0.01290477]
    epoch: 4, iter: 20, loss is: [0.02234135]
    epoch: 4, iter: 40, loss is: [0.02070921]
    epoch: 4, iter: 60, loss is: [0.01241408]
    epoch: 5, iter: 0, loss is: [0.02910613]
    epoch: 5, iter: 20, loss is: [0.01874577]
    epoch: 5, iter: 40, loss is: [0.01416044]
    epoch: 5, iter: 60, loss is: [0.09141838]
    epoch: 6, iter: 0, loss is: [0.02658093]
    epoch: 6, iter: 20, loss is: [0.02040408]
    epoch: 6, iter: 40, loss is: [0.03866465]
    epoch: 6, iter: 60, loss is: [0.02139562]
    epoch: 7, iter: 0, loss is: [0.02232074]
    epoch: 7, iter: 20, loss is: [0.03333586]
    epoch: 7, iter: 40, loss is: [0.02530864]
    epoch: 7, iter: 60, loss is: [0.01014854]
    epoch: 8, iter: 0, loss is: [0.01676147]
    epoch: 8, iter: 20, loss is: [0.02722653]
    epoch: 8, iter: 40, loss is: [0.01120919]
    epoch: 8, iter: 60, loss is: [0.00752459]
    epoch: 9, iter: 0, loss is: [0.01959429]
    epoch: 9, iter: 20, loss is: [0.03472038]
    epoch: 9, iter: 40, loss is: [0.02280031]
    epoch: 9, iter: 60, loss is: [0.01171026]


**两种方法的比较**
使用测试集计算两种方法的均方误差，比较两者的误差大小


```python
_,_,test_feature,test_label=data_load(data_path)
test_feature_1=modify_x(test_feature)
way1_out=net.forward(test_feature_1)

#转换为预测模式，更加节省内存速度更快性能更高
model.eval()
test_feature_2=paddle.to_tensor(test_feature)
way2_out=model(test_feature_2)


# way1_out=way1_out*(maximums[1]-minimums[1])+avgs[1]
# way2_out=way2_out*(maximums[1]-minimums[1])+avgs[1]
# test_label=test_label*(maximums[1]-minimums[1])+avgs[1]
test_label_2=paddle.to_tensor(test_label)
loss1=net.loss(way1_out,test_label)
loss2=F.square_error_cost(way2_out,label=test_label_2)
loss2=np.mean(loss2.numpy())

print(f'loss    way1:{loss1}  way2:{loss2}')
```

    loss    way1:0.04192041453699585  way2:0.02530328556895256



```python
way1_out=way1_out*(maximums[1]-minimums[1])+avgs[1]
way2_out=way2_out*(maximums[1]-minimums[1])+avgs[1]
test_label=test_label*(maximums[1]-minimums[1])+avgs[1]
test_label_2=paddle.to_tensor(test_label)
loss1=net.loss(way1_out,test_label)
loss2=F.square_error_cost(way2_out,label=test_label_2)
loss2=np.mean(loss2.numpy())

print(f'loss    way1:{loss1}  way2:{loss2}')
```

    loss    way1:135520.48384232534  way2:81800.5546875


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
