# Softmax

Softmax 函数一般用于多分类问题中，它是对逻辑斯蒂（logistic）回归的一种推广，也被称为多项逻辑斯蒂回归模型(multi-nominal logistic mode)。假设要实现 k 个类别的分类任务，Softmax 函数将输入数据 $x_i$ 映射到第 $i$ 个类别的概率 $y_i$ 如下计算：

$$
y_i=soft\max \left( x_i \right) =\frac{e^{x_i}}{\sum_{j=1}^k{e^{x_j}}}
$$

显然，$0<y_i<1$。图 1 给出了三类分类问题的  Softmax 输出示意图。在图中，对于取值为 4、1和-4 的 $x_1$、$x_2$ 和 $x_3$，通过 Softmax 变换后，将其映射到 (0,1) 之间的概率值。

<center><img src="https://raw.githubusercontent.com/lvjian0706/Deep-Learning-Img/master/Base/Softmax/img/softmax.png" width="500" hegiht="" ></center>
<center><br>图1：三类分类问题的softmax输出示意图</br></center>

由于 Softmax 输出结果的值累加起来为 1，因此可将输出概率最大的作为分类目标（图 1 中被分类为第一类）。

也可以从如下另外一个角度来理解图 1 中的内容：给定某个输入数据，可得到其分类为三个类别的初始结果，分别用 $x_1$、$x_2$ 和 $x_3$ 来表示。这三个初始分类结果分别是 4、1和-4。通过 Softmax 函数，得到了三个类别分类任务中以概率表示的更好的分类结果，即分别以 95.25%、4.71%和0.04% 归属于类别1、类别2 和类别3。显然，基于这样的概率值，可判断输入数据属于第一类。可见，通过使用 Softmax 函数，可求取输入数据在所有类别上的概率分布。