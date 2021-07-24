## DNN网络结构

DNN内部的神经网络层可以分为三类，输入层，隐藏层和输出层，如下图示例，一般来说第一层是输入层，最后一层是输出层，而中间的层数都是隐藏层。

![img](https://img-blog.csdnimg.cn/20191115162657992.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2doajc4NjExMA==,size_16,color_FFFFFF,t_70)

一般说到神经网络的层数是这样计算的，输入层不算，从隐藏层开始一直到输出层，一共有几层就代表着这是一个几层的神经网络，例如上图就是一个三层结构的神经网络。

层与层之间是全连接的，也就是说，第`i`层的任意一个神经元一定与第`i+1`层的任意一个神经元相连。虽然DNN看起来很复杂，但是从小的局部模型来说，还是和感知机一样，即一个线性关系
$$
z=\sum{w_ix_i+b}
$$
加上一个激活函数`𝜎(𝑧)`。

首先我们来看看线性关系系数w w*w*的定义。
以下图一个三层的DNN为例，第二层的第4个神经元到第三层的第2个神经元的线性系数定义为w 24 3 w_{24}^3*w*243。上标3代表线性系数w w*w*所在的层数，而下标对应的是输出的第三层索引2和输入的第二层索引4。
为什么不是w 42 3 w_{42}^3*w*423, 而是w 24 3 w_{24}^3*w*243呢？这主要是为了便于模型用于矩阵表示运算，如果是w 42 3 w_{42}^3*w*423而每次进行矩阵运算是w T x + b w^Tx+b*w**T**x*+*b*，需要进行转置。
将输出的索引放在前面的话，则线性运算不用转置，即直接为w x + b wx+b*w**x*+*b*。总结下，第𝑙−1层的第k个神经元到第𝑙层的第j个神经元的线性系数定义为w j k l w_{jk}^l*w**j**k**l*。注意，输入层是没有𝑤参数的。

![img](https://img-blog.csdnimg.cn/20191115162040589.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2doajc4NjExMA==,size_16,color_FFFFFF,t_70)

偏倚b b*b*类似于w w*w*。还是以这个三层的DNN为例，第二层的第三个神经元对应的偏倚定义为b 3 2 b^2_3*b*32。其中，上标2代表所在的层数，下标3代表偏倚所在的神经元的索引。同样的道理，第三个的第一个神经元的偏倚应该表示为b 1 3 b^3_1*b*13。同样的，输入层是没有偏倚参数b b*b*的。

![img](https://img-blog.csdnimg.cn/20191115162307898.png)

