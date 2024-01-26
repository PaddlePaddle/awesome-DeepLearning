# 深度学习基础知识 #

## 损失函数补充 ##

**损失函数**用来评价模型的**预测值**和**真实值**不一样的程度，损失函数越好，通常模型的性能越好。不同的模型用的损失函数一般也不一样。

**损失函数**分为**经验风险损失函数**和**结构风险损失函数**。经验风险损失函数指预测结果和实际结果的差别，结构风险损失函数是指经验风险损失函数加上正则项。

常见的损失函数以及其优缺点如下：

### 0-1损失函数(zero-one loss)

0-1损失是指预测值和目标值不相等为1， 否则为0:
$$
L(Y,f(X))=\left\{\begin{matrix}1,Y\neq f(X)
\\ 0,Y=f(X)
\end{matrix}\right.
$$
 特点：

(1)0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用.

(2)感知机就是用的这种损失函数。但是相等这个条件太过严格，因此可以放宽条件，即满足$|Y-f(x)|<T$时认为相等，
$$
L(Y,f(X))=\left\{\begin{matrix}1,|Y-f(x)|\geqslant T
\\ 0,|Y-f(x)|<  T

\end{matrix}\right.
$$

### 绝对值损失函数

绝对值损失函数是计算预测值与目标值的差的绝对值：
$$
L(Y,f(X))=|Y-f(x)|
$$

### log对数损失函数

**log对数损失函数**的标准形式如下：
$$
L(Y,P(Y|X))=-logP(Y|X)
$$
特点：

(1) log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。

(2)健壮性不强，相比于hinge loss对噪声更敏感。

(3)**逻辑回归**的损失函数就是log对数损失函数。

### 平方损失函数

平方损失函数标准形式如下：
$$
L(Y|f(X))=\sum_N(Y-f(X))^2
$$
特点：

(1)经常应用与回归问题

### 指数损失函数（exponential loss）

**指数损失函数**的标准形式如下：
$$
L(Y|f(X))=exp[-yf(x)]
$$
特点：

(1)对离群点、噪声非常敏感。经常用在AdaBoost算法中。

### Hinge 损失函数

Hinge损失函数标准形式如下：
$$
L(y,f(x))=max(0,1-yf(x))
$$
 特点：

(1)hinge损失函数表示如果被分类正确，损失为0，否则损失就为$1-yf(x)$。SVM就是使用这个损失函数。

(2)一般的$f(x)$是预测值，在-1到1之间，$y$是目标值(-1或1)。其含义是，$f(x)$的值在-1和+1之间就可以了，并不鼓励$|f(x)|>1$ ，即并不鼓励分类器过度自信，让某个正确分类的样本距离分割线超过1并不会有任何奖励，从而**使分类器可以更专注于整体的误差。**

(3) 健壮性相对较高，对异常点、噪声不敏感，但它没太好的概率解释。

### 感知损失(perceptron loss)函数

**感知损失函数**的标准形式如下：
$$
L(y,f(x))=max(0,-f(x))
$$
特点：

(1)是Hinge损失函数的一个变种，Hinge loss对判定边界附近的点(正确端)惩罚力度很高。而perceptron loss**只要样本的判定类别正确的话，它就满意，不管其判定边界的距离**。它比Hinge loss简单，因为不是max-margin boundary，所以模**型的泛化能力没 hinge loss强**。

### Mean Absolute Error (L1 Loss) 和 Mean Square Error（L2 Loss）

L1和L2 Loss很类似，而且一些性质比较着说更清楚，因此放在一起。首先计算方法上 L1 Loss 是求所有预测值和标签距离的平均数，L2 Loss 是求预测值和标签距离平方的平均数。二者的结构很类似，区别就是一个用了绝对值一个用了平方。在性质上，二者主要有三个方面的不同：离群点鲁棒性，梯度和是否可微。

## 池化方法补充

卷积神经网络(Convolution Neural Network, CNN)因其强大的特征提取能力而被广泛地应用到计算机视觉的各个领域，其中卷积层和池化层是组成CNN的两个主要部件。理论上来说，网络可以在不对原始输入图像执行降采样的操作，通过堆叠多个的卷积层来构建深度神经网络，如此一来便可以在保留更多空间细节信息的同时提取到更具有判别力的抽象特征。然而，考虑到计算机的算力瓶颈，通常都会引入池化层，来进一步地降低网络整体的计算代价，这是引入池化层最根本的目的。池化层大大降低了网络模型参数和计算成本，也在一定程度上降低了网络过拟合的风险。概括来说，池化层主要有以下五点作用：

- 增大网络感受野
- 抑制噪声，降低信息冗余
- 降低模型计算量，降低网络优化难度，防止网络过拟合
- 使模型对输入图像中的特征位置变化更加鲁棒

对于池化操作，大部分人第一想到的可能就是Max_Pooling和Average_Pooling，但实际上卷积神经网络的池化方法还有很多，本文将对业界目前所出现的一些池化方法进行归纳总结：

### 1. Max Pooling(最大池化)

**定义**

最大池化(Max Pooling)是将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。其定义如下：
$$
y_{kij}=\underset{(p,q)\in R_{ij}}{max}x_{kpq}
$$
其中，$y_{kij}$表示与第$k$个特征图有关的在矩形区域$R_{ij}$的最大池化输出值，$x_{kpq}$表示矩形区域中$R_{ij}$位于(p,q)处的元素。

对一个4![[公式]](https://www.zhihu.com/equation?tex=%5Ctimes)4的特征图邻域内的值，用一个2![[公式]](https://www.zhihu.com/equation?tex=%5Ctimes)2的filter，步长为2进行“扫描”，选择最大值输出到下一层，这叫做最大池化。

对于最大池化操作，只选择每个矩形区域中的最大值进入下一层，而其他元素将不会进入下一层。所以最大池化提取特征图中响应最强烈的部分进入下一层，这种方式摒弃了网络中大量的冗余信息，使得网络更容易被优化。同时这种操作方式也常常丢失了一些特征图中的细节信息，所以最大池化更多保留些图像的纹理信息。

### 2. Average Pooling(平均池化)

定义

平均池化(Average Pooling)是将输入的图像划分为若干个矩形区域，对每个子区域输出所有元素的平均值。其定义如下：

![[公式]](https://www.zhihu.com/equation?tex=y_%7Bk+i+j%7D%3D%5Cfrac%7B1%7D%7B%5Cleft%7C%5Cmathcal%7BR%7D_%7Bi+j%7D%5Cright%7C%7D+%5Csum_%7B%28p%2C+q%29+%5Cin+%5Cmathcal%7BR%7D_%7Bi+j%7D%7D+x_%7Bk+p+q%7D+%5C%5C)

其中，$y_{kij}$表示与第$k$个特征图有关的在矩形区域$R_{ij}$的平均池化输出值，$x_{kpq}$表示矩形区域$R_{ij}$中位于(p,q)处的元素，$R_{ij}$表示矩形区域$R_{ij}$中元素个数。

平均池化取每个矩形区域中的平均值，可以提取特征图中所有特征的信息进入下一层，而不像最大池化只保留值最大的特征，所以平均池化可以更多保留些图像的背景信息。

### 3.Global Average Pooling(全局平均池化)

定义

全局平均池化是一种特殊的平均池化，只不过它不划分若干矩形区域，而是将整个特征图中所有的元素取平均输出到下一层。其定义如下：

![[公式]](https://www.zhihu.com/equation?tex=y_%7Bk%7D%3D%5Cfrac%7B1%7D%7B%5Cleft%7C%5Cmathcal%7BR%7D%5Cright%7C%7D+%5Csum_%7B%28p%2C+q%29+%5Cin+%5Cmathcal%7BR%7D%7D+x_%7Bk+p+q%7D+%5C%5C)

其中，![[公式]](https://www.zhihu.com/equation?tex=y_%7Bk%7D)表示与第![[公式]](https://www.zhihu.com/equation?tex=k)个特征图的全局平均池化输出值，![[公式]](https://www.zhihu.com/equation?tex=x_%7Bk+p+q%7D)表示第![[公式]](https://www.zhihu.com/equation?tex=k)个特征图区域![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BR%7D)中位于(p,q)处的元素，![[公式]](https://www.zhihu.com/equation?tex=%7C%5Cmathcal%7BR%7D%7C)表示第![[公式]](https://www.zhihu.com/equation?tex=k)个特征图全部元素的个数。

对于一个输入特征图![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BX%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7Bh+%5Ctimes+w+%5Ctimes+d%7D),经过全局平均池化(GAP)之后生成新的特征图![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BO%7D+%5Cin+%5Cmathbb%7BR%7D%5E%7B1+%5Ctimes+1+%5Ctimes+d%7D)。

### 4. Mix Pooling(混合池化)

定义

为了提高训练较大CNN模型的正则化性能，受Dropout(将一半激活函数随机设置为0)的启发，Dingjun Yu等人提出了一种随机池化**Mix Pooling[2]**的方法，随机池化用随机过程代替了常规的确定性池化操作，在模型训练期间随机采用了最大池化和平均池化方法，并在一定程度上有助于防止网络过拟合现象。其定义如下：

![[公式]](https://www.zhihu.com/equation?tex=y_%7Bk+i+j%7D%3D%5Clambda+%5Ccdot+%5Cmax+_%7B%28p%2C+q%29+%5Cin+%5Cmathcal%7BR%7D_%7Bi+j%7D%7D+x_%7Bk+p+q%7D%2B%281-%5Clambda%29+%5Ccdot+%5Cfrac%7B1%7D%7B%5Cleft%7C%5Cmathcal%7BR%7D_%7Bi+j%7D%5Cright%7C%7D+%5Csum_%7B%28p%2C+q%29+%5Cin+%5Cmathcal%7BR%7D_%7Bi+j%7D%7D+x_%7Bk+p+q%7D+%5C%5C)

其中，![[公式]](https://www.zhihu.com/equation?tex=%5Clambda)是0或1的随机值，表示选择使用最大池化或平均池化，换句话说，混合池化以随机方式改变了池调节的规则，这将在一定程度上解决最大池和平均池所遇到的问题。

混合池化优于传统的最大池化和平均池化方法，并可以解决过拟合问题来提高分类精度。此外该方法所需要的计算开销可忽略不计，而无需任何超参数进行调整，可被广泛运用于CNN。

### 5. Stochastic Pooling(随机池化)

定义

随机池化**Stochastic Pooling[3]**是Zeiler等人于ICLR2013提出的一种池化操作。随机池化的计算过程如下：

- 先将方格中的元素同时除以它们的和sum，得到概率矩阵。
- 按照概率随机选中方格。
- pooling得到的值就是方格位置的值。

每个元素值表示对应位置处值的概率，现在只需要按照该概率来随机选一个，方法是：将其看作是9个变量的多项式分布，然后对该多项式分布采样即可，theano中有直接的multinomial()来函数完成。当然也可以自己用0-1均匀分布来采样，将单位长度1按照那9个概率值分成9个区间（概率越大，覆盖的区域越长，每个区间对应一个位置），然随机生成一个数后看它落在哪个区间。

随机池化只需对特征图中的元素按照其概率值大小随机选择，即元素值大的被选中的概率也大，而不像max-pooling那样，永远只取那个最大值元素，这使得随机池化具有更强的泛化能力。

### 6. Power Average Pooling(幂平均池化)

定义

幂平均池化**Power Average Pooling[5]**基于平均池化和最大池化的结合，它利用一个学习参数![[公式]](https://www.zhihu.com/equation?tex=p)来确定这两种方法的相对重要性；当![[公式]](https://www.zhihu.com/equation?tex=p%3D1)时，使用局部求和，当![[公式]](https://www.zhihu.com/equation?tex=p+%5Crightarrow+%5Cinfty)时，使用最大池化，其定义如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Ba%7D%7D%3D%5Csqrt%5Bp%5D%7B%5Csum_%7Bi+%5Cin+%5Cmathbf%7BR%7D%7D+%5Cmathbf%7Ba%7D_%7Bi%7D%5E%7Bp%7D%7D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BR%7D)表示待池化区域中的像素值集。

### 7. Detail-Preserving Pooling(DPP池化)

为了降低隐藏层的规模或数量，大多数CNN都会采用池化方式来减少参数数量，来改善某些失真的不变性并增加感受野的大小。由于池化本质上是一个有损的过程，所以每个这样的层都必须保留对网络可判别性最重要的部分进行激活。但普通的池化操作只是在特征图区域内进行简单的平均或最大池化来进行下采样过程，这对网络的精度有比较大的影响。基于以上几点，Faraz Saeedan等人提出一种自适应的池化方法-DPP池化**Detail-Preserving Pooling[6]**，该池化可以放大空间变化并保留重要的图像结构细节，且其内部的参数可通过反向传播加以学习。DPP池化主要受**Detail-Preserving Image Downscaling[7]**的启发。

- Detail-Preserving Image Downscaling

![[公式]](https://www.zhihu.com/equation?tex=O%5Bp%5D%3D%5Cfrac%7B1%7D%7Bk_%7Bp%7D%7D+%5Csum_%7Bq+%5Cin+%5COmega_%7Bp%7D%7D+I%5Bq%5D+%5Ccdot%5C%7CI%5Bq%5D-%5Ctilde%7BI%7D%5Bp%5D%5C%7C%5E%7B%5Clambda%7D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=I)是原图，![[公式]](https://www.zhihu.com/equation?tex=O)是output，[ ]表示取对于坐标像素值。

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BI%7D%3DI_%7BD%7D+%2A+%5Cfrac%7B1%7D%7B16%7D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D+1+%26+2+%26+1+%5Cend%7Barray%7D%5Cright%5D%5E%7BT%7D%5Cleft%5B%5Cbegin%7Barray%7D%7Blll%7D+1+%26+2+%26+1+%5Cend%7Barray%7D%5Cright%5D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=ID)是施加到输入随后的下采样，其随后由一个近似的二维高斯滤波器平滑化的箱式滤波器的结果。 

DPID的滤波图，与普通双边滤波器不同，它奖励输入强度的差异，使得与![[公式]](https://www.zhihu.com/equation?tex=I)的差异较大的像素值贡献更大。



- Detail-Preserving Pooling

a. 将上部分中的![[公式]](https://www.zhihu.com/equation?tex=L2+Norm)替换成一个可学习的generic scalar reward function：

![[公式]](https://www.zhihu.com/equation?tex=D_%7B%5Calpha%2C+%5Clambda%7D%28I%29%5Bp%5D%3D%5Cfrac%7B1%7D%7B%5Csum_%7Bq%5E%7B%5Cprime%7D+%5Cin+%5COmega_%7Bp%7D%7D+%5Comega_%7B%5Calpha%2C+%5Clambda%5Cleft%5Bp%2C+q%5E%7B%5Cprime%7D%5Cright%5D%7D%7D+%5Csum_%7Bq+%5Cin+%5COmega_%7Bp%7D%7D+%5Comega_%7B%5Calpha%2C+%5Clambda%7D%5Bp%2C+q%5D+I%5Bq%5D+%5C%5C)

b. 首先给出weight的表示：

![[公式]](https://www.zhihu.com/equation?tex=%5Comega_%7B%5Calpha%2C+%5Clambda%7D%5Bp%2C+q%5D%3D%5Calpha%2B%5Crho_%7B%5Clambda%7D%28I%5Bq%5D-%5Ctilde%7BI%7D%5Bp%5D%29+%5C%5C)

c. 这里给出了两种reward function：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Crho_%7Bs+y+m%7D%28x%29+%26%3D%5Cleft%28%5Csqrt%7Bx%5E%7B2%7D%2B%5Cvarepsilon%5E%7B2%7D%7D%5Cright%29%5E%7B%5Clambda%7D+%5C%5C+%5Crho_%7BA+s+y+m%7D%28x%29+%26%3D%5Cleft%28%5Csqrt%7B%5Cmax+%280%2C+x%29%5E%7B2%7D%2B%5Cvarepsilon%5E%7B2%7D%7D%5Cright%29%5E%7B%5Clambda%7D+%5Cend%7Baligned%7D+%5C%5C)

d. 作者又补充了的生成：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7BI%7D_%7BF%7D%5Bp%5D%3D%5Csum_%7Bq+%5Cin+%5Ctilde%7B%5COmega%7D_%7Bp%7D%7D+F%5Bq%5D+I%5Bq%5D+%5C%5C)

### 8. Local Importance Pooling(局部重要性池化)

CNN通常使用空间下采样层来缩小特征图，以实现更大的接受场和更少的内存消耗，但对于某些任务而言，这些层可能由于不合适的池化策略而丢失一些重要细节，最终损失模型精度。为此，作者从局部重要性的角度提出了局部重要性池化**Local Importance Pooling[8]**，通过基于输入学习自适应重要性权重，LIP可以在下采样过程中自动增加特征判别功能。

### **定义**

- 池化操作可归纳为如下公式：

![[公式]](https://www.zhihu.com/equation?tex=O_%7Bx%5E%7B%5Cprime%7D%2C+y%5E%7B%5Cprime%7D%7D%3D%5Cfrac%7B%5Csum_%7B%28%5CDelta+x%2C+%5CDelta+y%29+%5Cin+%5COmega%7D+F%28I%29_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D+I_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D%7D%7B%5Csum_%7B%28%5CDelta+x%2C+%5CDelta+y%29+%5Cin+%5COmega%7D+F%28I%29_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D%7D+%5C%5C)

其中![[公式]](https://www.zhihu.com/equation?tex=F)的大小和特征![[公式]](https://www.zhihu.com/equation?tex=I)一致，代表每个点的重要性。

首先最大池化对应的最大值不一定是最具区分力的特征，并且在梯度更新中也难以更新到最具区分力的特征，除非最大值被抑制掉。而步长为2的卷积问题主要在于固定的采样位置。

因此，合适的池化操作应该包含两点：
a. 下采样的位置要尽可能非固定间隔
b. 重要性的函数![[公式]](https://www.zhihu.com/equation?tex=F)需通过学习获得

- Local Importance-based Pooling



LIP首先在原特征图上学习一个类似于注意力的特征图，然后再和原特征图进行加权求均值，公式可表述如下：

![[公式]](https://www.zhihu.com/equation?tex=O_%7Bx%5E%7B%5Cprime%7D%2C+y%5E%7B%5Cprime%7D%7D%3D%5Cfrac%7B%5Csum_%7B%28%5CDelta+x%2C+%5CDelta+y%29+%5Cin+%5COmega%7D+I_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D+%5Cexp+%28%5Cmathcal%7BG%7D%28I%29%29_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D%7D%7B%5Csum_%7B%28%5CDelta+x%2C+%5CDelta+y%29+%5Cin+%5COmega%7D+%5Cexp+%28%5Cmathcal%7BG%7D%28I%29%29_%7Bx%2B%5CDelta+x%2C+y%2B%5CDelta+y%7D%7D+%5C%5C)

对于![[公式]](https://www.zhihu.com/equation?tex=G)函数，可以通过如下图d和e两种方式实现(分别称之为Projection和Bottleneck-X)。而对应的ResNet-LIP则如下图b所示：



### 9. Soft Pooling(软池化)

现有的一些池化方法大都基于最大池化和平均池化的不同组合，而软池化**Soft Pooling[9]**是基于softmax加权的方法来保留输入的基本属性，同时放大更大强度的特征激活。与maxpooling不同，softpool是可微的，所以网络在反向传播过程中为每个输入获得一个梯度，这有利于提高训练效果。

定义

- SoftPool的计算流程如下：
  a. 特征图透过滑动视窗来框选局部数值
  b. 框选的局部数值会先经过指数计算，计算出的值为对应的特征数值的权重
  c. 将各自的特征数值与其相对应的权重相乘
  d. 最后进行加总

这样的方式让整体的局部数值都有所贡献，重要的特征占有较高的权重。比Max pooling(直接选择最大值)、Average pooling (求平均，降低整个局部的特征强度) 能够保留更多讯息。

- SoftPool的数学定义如下：

计算特征数值的权重，其中R为框选的局部区域，a为特征数值

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bw%7D_%7Bi%7D%3D%5Cfrac%7Be%5E%7B%5Cmathbf%7Ba%7D_%7Bi%7D%7D%7D%7B%5Csum_%7Bj+%5Cin+%5Cmathbf%7BR%7D%7D+e%5E%7B%5Cmathbf%7Ba%7D_%7Bj%7D%7D%7D+%5C%5C)

将相应的特征数值与权重相乘后做加总操作

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7B%5Cmathbf%7Ba%7D%7D%3D%5Csum_%7Bi+%5Cin+%5Cmathbf%7BR%7D%7D+%5Cmathbf%7Bw%7D_%7Bi%7D+%2A+%5Cmathbf%7Ba%7D_%7Bi%7D+%5C%5C)

- 梯度计算: 下图可以很清楚的指导使用SoftPool的Gradient计算流程。与Max Pooling不同，SoftPool是可微的，因此在反向传播至少会分配一个最小梯度值进行更新。

作为一种新颖地池化方法，SoftPool可以在保持池化层功能的同时尽可能减少池化过程中带来的信息损失，更好地保留信息特征并因此改善CNN中的分类性能。大量的实验结果表明该算法的性能优于原始的Avg池化与Max池化。随着神经网络的设计变得越来越困难，而通过NAS等方法也几乎不能大幅度提升算法的性能，为了打破这个瓶颈，从基础的网络层优化入手，不失为一种可靠有效的精度提升手段。

## 数据增强方法

### 1 翻转（Flip）

可以对图片进行水平和垂直翻转。一些框架不提供垂直翻转功能。但是，一个垂直反转的图片等同于图片的180度旋转，然后再执行水平翻转。

你可以使用你喜欢的工具包进行下面的任意命令进行翻转，数据增强因子=2或4

```text
# NumPy.'img' = A single image.
flip_1 = np.fliplr(img)
# TensorFlow. 'x' = A placeholder for an image.
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
flip_2 = tf.image.flip_up_down(x)
flip_3 = tf.image.flip_left_right(x)
flip_4 = tf.image.random_flip_up_down(x)
flip_5 = tf.image.random_flip_left_right(x)
```

### 2 旋转（Rotation）

一个关键性的问题是当旋转之后图像的维数可能并不能保持跟原来一样。如果你的图片是正方形的，那么以直角旋转将会保持图像大小。如果它是长方形，那么180度的旋转将会保持原来的大小。以更精细的角度旋转图像也会改变最终的图像尺寸。我们将在下一节中看到我们如何处理这个问题。

你可以使用你喜欢的工具包执行以下的旋转命令。数据增强因子= 2或4。

```python3
# Placeholders: 'x' = A single image, 'y' = A batch of images
# 'k' denotes the number of 90 degree anticlockwise rotations
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
rot_90 = tf.image.rot90(img, k=1)
rot_180 = tf.image.rot90(img, k=2)
# To rotate in any angle. In the example below, 'angles' is in radians
shape = [batch, height, width, 3]
y = tf.placeholder(dtype = tf.float32, shape = shape)
rot_tf_180 = tf.contrib.image.rotate(y, angles=3.1415)
# Scikit-Image. 'angle' = Degrees. 'img' = Input Image
# For details about 'mode', checkout the interpolation section below.
rot = skimage.transform.rotate(img, angle=45, mode='reflect')
```

### 3 缩放比例（Scale）

图像可以向外或向内缩放。向外缩放时，最终图像尺寸将大于原始图像尺寸。大多数图像框架从新图像中剪切出一个部分，其大小等于原始图像。我们将在下一节中处理向内缩放，因为它会缩小图像大小，迫使我们对超出边界的内容做出假设。

您可以使用scikit-image使用以下命令执行缩放。数据增强因子=任意。

```text
# Scikit Image. 'img' = Input Image, 'scale' = Scale factor
# For details about 'mode', checkout the interpolation section below.
scale_out = skimage.transform.rescale(img, scale=2.0, mode='constant')
scale_in = skimage.transform.rescale(img, scale=0.5, mode='constant')
# Don't forget to crop the images back to the original size (for 
# scale_out)
```

### 4 裁剪（Crop）

与缩放不同，我们只是从原始图像中随机抽样一个部分。然后，我们将此部分的大小调整为原始图像大小。这种方法通常称为随机裁剪。你可以使用以下任何TensorFlow命令执行随机裁剪。

```text
# TensorFlow. 'x' = A placeholder for an image.
original_size = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = original_size)
# Use the following commands to perform random crops
crop_size = [new_height, new_width, channels]
seed = np.random.randint(1234)
x = tf.random_crop(x, size = crop_size, seed = seed)
output = tf.images.resize_images(x, size = original_size)
```

### 5 移位（Translation）

移位只涉及沿X或Y方向（或两者）移动图像。在下面的示例中，我们假设图像在其边界之外具有黑色背景，并且被适当地移位。这种增强方法非常有用，因为大多数对象几乎可以位于图像的任何位置。这迫使你的卷积神经网络看到所有角落。你可以使用以下命令在TensorFlow中执行转换。数据增强因子=任意。

```text
# pad_left, pad_right, pad_top, pad_bottom denote the pixel 
# displacement. Set one of them to the desired value and rest to 0
shape = [batch, height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
# We use two functions to get our desired augmentation
x = tf.image.pad_to_bounding_box(x, pad_top, pad_left, height + pad_bottom + pad_top, width + pad_right + pad_left)
output = tf.image.crop_to_bounding_box(x, pad_bottom, pad_right, height, width)
```

### 6 高斯噪声（Gaussian Noise）

当您的神经网络试图学习可能无用的高频特征（大量出现的模式）时，通常会发生过度拟合。具有零均值的高斯噪声基本上在所有频率中具有数据点，从而有效地扭曲高频特征。这也意味着较低频率的组件（通常是您的预期数据）也会失真，但你的神经网络可以学会超越它。添加适量的噪音可以增强学习能力。

一个色调较低的版本是盐和胡椒噪音，它表现为随机的黑白像素在图像中传播。这类似于通过向图像添加高斯噪声而产生的效果，但可能具有较低的信息失真水平。

```text
#TensorFlow. 'x' = A placeholder for an image.
shape = [height, width, channels]
x = tf.placeholder(dtype = tf.float32, shape = shape)
# Adding Gaussian noise
noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
dtype=tf.float32)
output = tf.add(x, noise)
```

### 7. 高级增强技术

现实世界中，自然数据仍然可以存在于上述简单方法无法解释的各种条件下。例如，让我们承担识别照片中景观的任务。景观可以是任何东西：冻结苔原，草原，森林等。听起来像一个非常直接的分类任务吧？除了一件事，你是对的。我们忽略了影响照片表现中的一个重要特征 - 拍摄照片的季节。

如果我们的神经网络不了解某些景观可以在各种条件下（雪，潮湿，明亮等）存在的事实，它可能会将冰冻的湖岸虚假地标记为冰川或湿地作为沼泽。

缓解这种情况的一种方法是添加更多图片，以便我们考虑所有季节性变化。但这是一项艰巨的任务。扩展我们的数据增强概念，想象一下人工生成不同季节的效果有多酷？

在没有进入血腥细节的情况下，条件GAN可以将图像从一个域转换为图像到另一个域。

上述方法是稳健的，但计算密集。更便宜的替代品将被称为神经风格转移（neural style transfer）。它抓取一个图像（又称“风格”）的纹理、氛围、外观，并将其与另一个图像的内容混合。使用这种强大的技术，我们产生类似于条件GAN的效果（事实上，这种方法是在cGAN发明之前引入的！）。

### 8. 插值方法

如果您想要翻译不具有黑色背景的图像，该怎么办？如果你想向内扩展怎么办？或者以更精细的角度旋转？在我们执行这些转换后，我们需要保留原始图像大小。由于我们的图像没有关于其边界之外的任何信息，我们需要做出一些假设。通常，假设图像边界之外的空间在每个点都是常数0。因此，当您进行这些转换时，会得到一个未定义图像的黑色区域。

#### 常数（Constant）

最简单的插值方法是用一些常数值填充未知区域。这可能不适用于自然图像，但可以用于在单色背景下拍摄的图像。

#### 边界（Edge）

在边界之后扩展图像的边缘值。此方法适用于温和移位。

#### 反射（Reflect）

图像像素值沿图像边界反射。此方法适用于包含树木，山脉等的连续或自然背景。

#### 对称（Symmetric）

该方法类似于反射，除了在反射边界处制作边缘像素的副本的事实。通常，反射和对称可以互换使用，但在处理非常小的图像或图案时会出现差异。

#### 包裹（Wrap）

图像只是重复超出其边界，就好像它正在平铺一样。这种方法并不像其他方法那样普遍使用，因为它对很多场景都没有意义。

除此之外，你可以设计自己的方法来处理未定义的空间，但通常这些方法对大多数分类问题都可以。



## 图像分类综述

### 一、图像分类介绍

　　什么是图像分类，核心是从给定的分类集合中给图像分配一个标签的任务。实际上，这意味着我们的任务是分析一个输入图像并返回一个将图像分类的标签。标签来自预定义的可能类别集。

　　示例：我们假定一个可能的类别集categories = {dog, cat, eagle}，之后我们提供一张图1给分类系统：

![image](https://z3.ax1x.com/2021/07/09/RxDOgS.png)

　　这里的目标是根据输入图像，从类别集中分配一个类别，这里为dog,我们的分类系统也可以根据概率给图像分配多个标签，如dog:95%，cat:4%，eagle:1%。

　　图像分类的任务就是给定一个图像，正确给出该图像所属的类别。对于超级强大的人类视觉系统来说，判别出一个图像的类别是件很容易的事，但是对于计算机来说，并不能像人眼那样一下获得图像的语义信息。

　　计算机能看到的只是一个个像素的数值，对于一个RGB图像来说，假设图像的尺寸是32\*32，那么机器看到的就是一个形状为3\*32\*32的矩阵，或者更正式地称其为“张量”（“张量”简单来说就是高维的矩阵），那么机器的任务其实也就是寻找一个函数关系，这个函数关系能够将这些像素的数值映射到一个具体的类别（类别可以用某个数值表示）。

### 二、应用场景

　　图像分类更适用于图像中待分类的物体是单一的，如上图1中待分类物体是单一的，如果图像中包含多个目标物，如下图3，可以使用多标签分类或者目标检测算法。

### 三、传统图像分类算法

　　通常完整建立图像识别模型一般包括底层特征学习、特征编码、空间约束、分类器设计、模型融合等几个阶段，如图4所示。

![image](https://z3.ax1x.com/2021/07/09/RxrC40.png)

　　**1).** **底层特征提取**: 通常从图像中按照固定步长、尺度提取大量局部特征描述。常用的局部特征包括SIFT(Scale-Invariant Feature Transform, 尺度不变特征转换) 、HOG(Histogram of Oriented Gradient, 方向梯度直方图) 、LBP(Local Bianray Pattern, 局部二值模式)等，一般也采用多种特征描述，防止丢失过多的有用信息。

　　**2).** **特征编码**: 底层特征中包含了大量冗余与噪声，为了提高特征表达的鲁棒性，需要使用一种特征变换算法对底层特征进行编码，称作特征编码。常用的特征编码方法包括向量量化编码、稀疏编码、局部线性约束编码、Fisher向量编码等。

　　**3).** **空间特征约束**: 特征编码之后一般会经过空间特征约束，也称作特征汇聚。特征汇聚是指在一个空间范围内，对每一维特征取最大值或者平均值，可以获得一定特征不变形的特征表达。金字塔特征匹配是一种常用的特征汇聚方法，这种方法提出将图像均匀分块，在分块内做特征汇聚。

　　**4).** **通过分类器分类**: 经过前面步骤之后一张图像可以用一个固定维度的向量进行描述，接下来就是经过分类器对图像进行分类。通常使用的分类器包括SVM(Support Vector Machine, 支持向量机)、随机森林等。而使用核方法的SVM是最为广泛的分类器，在传统图像分类任务上性能很好。

　　这种传统的图像分类方法在PASCAL VOC竞赛中的图像分类算法中被广泛使用 。

### 四、深度学习算法

　　Alex Krizhevsky在2012年ILSVRC提出的CNN模型取得了历史性的突破，效果大幅度超越传统方法，获得了ILSVRC2012冠军，该模型被称作AlexNet。这也是首次将深度学习用于大规模图像分类中。

　　从AlexNet之后，涌现了一系列CNN模型，不断地在ImageNet上刷新成绩，如图5展示。随着模型变得越来越深以及精妙的结构设计，Top-5的错误率也越来越低，降到了3.5%附近。而在同样的ImageNet数据集上，人眼的辨识错误率大概在5.1%，也就是目前的深度学习模型的识别能力已经超过了人眼。

#### 1、CNN

　　传统CNN包含卷积层、全连接层等组件，并采用softmax多类别分类器和多类交叉熵损失函数，一个典型的卷积神经网络如图6所示，我们先介绍用来构造CNN的常见组件。

![image](https://z3.ax1x.com/2021/07/09/RxsmQS.png)

- 卷积层(convolution layer): 执行卷积操作提取底层到高层的特征，发掘出图片局部关联性质和空间不变性质。

- 池化层(pooling layer): 执行降采样操作。通过取卷积输出特征图中局部区块的最大值(max-pooling)或者均值(avg-pooling)。降采样也是图像处理中常见的一种操作，可以过滤掉一些不重要的高频信息。

- 全连接层(fully-connected layer，或者fc layer): 输入层到隐藏层的神经元是全部连接的。
- 非线性变化: 卷积层、全连接层后面一般都会接非线性变化函数，例如Sigmoid、Tanh、ReLu等来增强网络的表达能力，在CNN里最常使用的为ReLu激活函数。
- Dropout: 在模型训练阶段随机让一些隐层节点权重不工作，提高网络的泛化能力，一定程度上防止过拟合。

　　另外，在训练过程中由于每层参数不断更新，会导致下一次输入分布发生变化，这样导致训练过程需要精心设计超参数。如2015年Sergey Ioffe和Christian Szegedy提出了Batch Normalization (BN)算法 中，每个batch对网络中的每一层特征都做归一化，使得每层分布相对稳定。BN算法不仅起到一定的正则作用，而且弱化了一些超参数的设计。

　　经过实验证明，BN算法加速了模型收敛过程，在后来较深的模型中被广泛使用。

#### 2、VGG

　　牛津大学VGG(Visual Geometry Group)组在2014年ILSVRC提出的模型被称作VGG模型。该模型相比以往模型进一步加宽和加深了网络结构，它的核心是五组卷积操作，每两组之间做Max-Pooling空间降维。同一组内采用多次连续的3X3卷积，卷积核的数目由较浅组的64增多到最深组的512，同一组内的卷积核数目是一样的。卷积之后接两层全连接层，之后是分类层。

　　由于每组内卷积层的不同，有11、13、16、19层这几种模型，下图展示一个16层的网络结构。VGG模型结构相对简洁，提出之后也有很多文章基于此模型进行研究，如在ImageNet上首次公开超过人眼识别的模型就是借鉴VGG模型的结构。

![image](https://z3.ax1x.com/2021/07/09/Rxs6SK.png)

#### 3、GoogLeNet

　　GoogLeNet 在2014年ILSVRC的获得了冠军，在介绍该模型之前我们先来了解NIN(Network in Network)模型和Inception模块，因为GoogLeNet模型由多组Inception模块组成，模型设计借鉴了NIN的一些思想。

　　NIN模型主要有两个特点：

1. 引入了多层感知卷积网络(Multi-Layer Perceptron Convolution, MLPconv)代替一层线性卷积网络。MLPconv是一个微小的多层卷积网络，即在线性卷积后面增加若干层1x1的卷积，这样可以提取出高度非线性特征。
2. 传统的CNN最后几层一般都是全连接层，参数较多。而NIN模型设计最后一层卷积层包含类别维度大小的特征图，然后采用全局均值池化(Avg-Pooling)替代全连接层，得到类别维度大小的向量，再进行分类。这种替代全连接层的方式有利于减少参数。

　　Inception模块如下图8所示，下图左是最简单的设计，输出是3个卷积层和一个池化层的特征拼接。这种设计的缺点是池化层不会改变特征通道数，拼接后会导致特征的通道数较大，经过几层这样的模块堆积后，通道数会越来越大，导致参数和计算量也随之增大。

　　为了改善这个缺点，下图右引入3个1x1卷积层进行降维，所谓的降维就是减少通道数，同时如NIN模型中提到的1x1卷积也可以修正线性特征。

![image](https://z3.ax1x.com/2021/07/09/RxsIYt.png)

　　GoogLeNet由多组Inception模块堆积而成。另外，在网络最后也没有采用传统的多层全连接层，而是像NIN网络一样采用了均值池化层；但与NIN不同的是，GoogLeNet在池化层后加了一个全连接层来映射类别数。

　　除了这两个特点之外，由于网络中间层特征也很有判别性，GoogLeNet在中间层添加了两个辅助分类器，在后向传播中增强梯度并且增强正则化，而整个网络的损失函数是这个三个分类器的损失加权求和。

　　GoogLeNet整体网络结构如图9所示，总共22层网络：开始由3层普通的卷积组成；接下来由三组子网络组成，第一组子网络包含2个Inception模块，第二组包含5个Inception模块，第三组包含2个Inception模块；然后接均值池化层、全连接层。

![image](https://z3.ax1x.com/2021/07/09/Rxs7Sf.png)

　　上面介绍的是GoogLeNet第一版模型(称作GoogLeNet-v1)。GoogLeNet-v2引入BN层；GoogLeNet-v3 对一些卷积层做了分解，进一步提高网络非线性能力和加深网络；GoogLeNet-v4引入下面要讲的ResNet设计思路。从v1到v4每一版的改进都会带来准确度的提升，介于篇幅，这里不再详细介绍v2到v4的结构。

#### 4、ResNet

　　ResNet(Residual Network) 是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。针对随着网络训练加深导致准确度下降的问题，ResNet提出了残差学习方法来减轻训练深层网络的困难。

　　在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含两条路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。

　　残差模块如图10所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。

![image](https://z3.ax1x.com/2021/07/09/Rxsb6S.png)

　　图11展示了50、101、152层网络连接示意图，使用的是瓶颈模块。这三个模型的区别在于每组中残差模块的重复次数不同(见图右上角)。ResNet训练收敛较快，成功的训练了上百乃至近千层的卷积神经网络。

![img](https://img2020.cnblogs.com/i-beta/1126989/202003/1126989-20200311151937565-832752578.png)
