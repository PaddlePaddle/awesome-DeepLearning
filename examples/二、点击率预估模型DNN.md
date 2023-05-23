DNN原理

DNN的结构：

DNN可以理解为有很多隐藏层的神经网络。这个很多其实也没有什么度量标准, 多层神经网络和深度神经网络DNN其实也是指的一个东西，当然，DNN有时也叫做多层感知机（Multi-Layer perceptron,MLP）, 名字实在是多。后面我们讲到的神经网络都默认为DNN。

从DNN按不同层的位置划分，DNN内部的神经网络层可以分为三类，输入层，隐藏层和输出层,如下图示例，一般来说第一层是输入层，最后一层是输出层，而中间的层数都是隐藏层。

![1042406-20170220122323148-1704308672](C:\Users\apple\Desktop\image\1042406-20170220122323148-1704308672.png)

层与层之间是全连接的，也就是说，第i层的任意一个神经元一定与第i+1层的任意一个神经元相连。虽然DNN看起来很复杂，但是从小的局部模型来说，还是和感知机一样，即一个线性关系 ![equation (5)](C:\Users\apple\Desktop\image\equation (5).svg)加上一个激活函数 ![equation (6)](C:\Users\apple\Desktop\image\equation (6).svg)。

由于DNN层数多，则我们的线性关系系数w和偏倚b的数量也就是很多了。具体的参数在DNN是如何定义的呢？

首先看线性关系系数w的定义。以下图一个三层的DNN为例，第二层的第4个神经元到第三层的第2个神经元的线性关系定义为![equation](C:\Users\apple\Desktop\image\equation.svg)上标3代表线性系数w所在的层数，而下标对应的是输出的第三层索引2和输入的第二层索引4。你也许会问，为什么不是![equation (1)](C:\Users\apple\Desktop\image\equation (1).svg)呢？这主要是为了便于模型用于矩阵表示运算，如果是![equation (1)](C:\Users\apple\Desktop\image\equation (1).svg)而每次进行矩阵运算是![equation (2)](C:\Users\apple\Desktop\image\equation (2).svg)，需要进行转置。将输出的索引放在前面的话，则线性运算不用转置，即直接为 ![equation (3)](C:\Users\apple\Desktop\image\equation (3).svg)。第l−1层的第k个神经元到第l层的第j个神经元的线性系数定义为![equation (4)](C:\Users\apple\Desktop\image\equation (4).svg)。注意，输入层是没有w参数的。

![v2-64f27adb53a7a8f142b40462ac977c61_720w](C:\Users\apple\Desktop\image\v2-64f27adb53a7a8f142b40462ac977c61_720w.png)

再看偏倚b的定义。还是以这个三层的DNN为例，第二层的第三个神经元对应的偏倚定义为![equation (7)](C:\Users\apple\Desktop\image\equation (7).svg) .其中，上标2代表所在的层数，下标3代表偏倚所在的神经元的索引。同样的道理，第三层的第一个神经元的偏倚应该表示为![equation (8)](C:\Users\apple\Desktop\image\equation (8).svg).输出层是没有偏倚参数的。

![v2-21fc3ed328124c126b9db46f766524b4_720w](C:\Users\apple\Desktop\image\v2-21fc3ed328124c126b9db46f766524b4_720w.png)

DNN的前向传播算法：

假设选择的激活函数是 ![equation (6)](C:\Users\apple\Desktop\image\equation (6).svg) ，隐藏层和输出层的输出值为**a**，则对于下图的三层DNN,利用和感知机一样的思路，我们可以利用上一层的输出计算下一层的输出，也就是所谓的DNN前向传播算法。

![v2-fe629aacecd2632fe375ea3b4e8184e7_720w](C:\Users\apple\Desktop\image\v2-fe629aacecd2632fe375ea3b4e8184e7_720w.jpg)

所谓的DNN前向传播算法就是利用若干个权重系数矩阵W,偏倚向量b来和输入值向量x进行一系列线性运算和激活运算，从输入层开始，一层层的向后计算，一直到运算到输出层，得到输出结果为值。

输入: 总层数L，所有隐藏层和输出层对应的矩阵W,偏倚向量b，输入值向量x

输出：输出层的输出。

DNN的后向传播算法：

BP算法的目的就是为了寻找合适的W使得损失函数Loss达到某一个比较小的值（极小值)。

在DNN中，损失函数优化极值求解的过程最常见的一般是通过梯度下降法来一步步迭代完成的，当然也有其他的方法。

输入：总层数L，以及各隐藏层与输出层的神经元个数，激活函数，损失函数，迭代步长a，最大迭代次数max，迭代阈值ε ，输入的m个训练样本

输出：各隐藏层与输出层的线性关系系数矩阵W和偏倚向量。

DNN算法流程：

由于梯度下降法有批量（Batch)，小批量(mini-Batch)，随机三个变种, 为了简化描述, 这里我们以最基本的批量梯度下降法为例来描述反向传播算法。实际上在业界使用最多的是mini-Batch的 梯度下降法。不过区别又仅在于迭代时训练样本的选择而已。

输入: 总层数![[公式]](https://www.zhihu.com/equation?tex=L), 以及各隐藏层与输出层的神经元个数, 激活函数, 损失函数, 选代步长 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha),最大迭代次数![[公式]](https://www.zhihu.com/equation?tex=MAX)与停止迭代阈值![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon), 输入的![[公式]](https://www.zhihu.com/equation?tex=m)个训练样本 ![[公式]](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B%5Cleft%28x_%7B1%7D%2C+y_%7B1%7D%5Cright%29%2C%5Cleft%28x_%7B2%7D%2C+y_%7B2%7D%5Cright%29%2C+%5Cldots%2C%5Cleft%28x_%7Bm%7D%2C+y_%7Bm%7D%5Cright%29%5Cright%5C%7D)。

输出: 各隐藏层与输出层的线性关系系数矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W) 和偏倚向量![[公式]](https://www.zhihu.com/equation?tex=b)

1. 初始化各隐藏层与输出层的线性关系系数矩阵![[公式]](https://www.zhihu.com/equation?tex=W)和偏倚向量![[公式]](https://www.zhihu.com/equation?tex=b)的值为一个随机值。

2. for iter to 1 to MAX:
   2.1 for i=1 to m :
       a. 将DNN输入![equation (9)](C:\Users\apple\Desktop\image\equation (9).svg) 设置为 ![equation (10)](C:\Users\apple\Desktop\image\equation (10).svg)

   ​    b. for l=L-1 to 2，进行前向传播算法计算![equation (11)](C:\Users\apple\Desktop\image\equation (11).svg) 

   ​    c. 通过损失函数计算输出层的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%5E%7Bi%2C+L%7D)  d. for ![[公式]](https://www.zhihu.com/equation?tex=l%3D) L-1 to 2 , 进行反向传播算法计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdelta%5E%7Bi%2C+l%7D%3D%5Cleft%28W%5E%7Bl%2B1%7D%5Cright%29%5E%7BT%7D+%5Cdelta%5E%7Bi%2C+l%2B1%7D+%5Codot+%5Csigma%5E%7B%5Cprime%7D%5Cleft%28z%5E%7Bi%2C+l%7D%5Cright%29) 

   2.2 for l=2 to L， 更新第l层的![equation (12)](C:\Users\apple\Desktop\image\equation (12).svg)

![equation (13)](C:\Users\apple\Desktop\image\equation (13).svg)

​       2-3. 如果所有![equation (14)](C:\Users\apple\Desktop\image\equation (14).svg)的变化值都小于停止迭代阈值ε， 则跳出迭代循环到步骤3。

        3. 输出各隐藏层与输出层的线性关系系数![equation (15)](C:\Users\apple\Desktop\image\equation (15).svg)和矩阵偏倚向量![equation (16)](C:\Users\apple\Desktop\image\equation (16).svg)。