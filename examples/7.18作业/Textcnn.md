## Textcnn

![](C:\Users\jasonzhao\jason\awesome-DeepLearning\examples\7.18作业\textcnn.png)

文本中的textcnn 与图像中卷积神经网络结构一致


TextCNN包含四部分：词嵌入、卷积、池化、全连接+softmax
Embedding：第一层是图中最左边的7乘5的句子矩阵，每行是词向量，维度=5，这个可以类比为图像中的原始像素点。
Convolution：然后经过 kernel_sizes=(2,3,4) 的一维卷积层，每个kernel_size 有两个输出 channel。
MaxPolling：第三层是一个1-max pooling层，这样不同长度句子经过pooling层之后都能变成定长的表示。
FullConnection and Softmax：最后接一层全连接的 softmax 层，输出每个类别的概率。

### 卷积

由于不同的论文对卷积的表达是不同的，所以以上图为例。上图中宽代表词向量，高代表不同的词。

由于TextCNN采用的卷积是一维卷积，所以卷积核的宽度和词嵌入的维度是一致的。而卷积核的高度h代表的是每次窗口取的词数。

对于每一次滑窗的结果
$$
c_i=f(w*x_{i:i+h-1})+b
$$
f是非线性函数，一般选用relu或者tanh

由于卷积运算是对应元素相乘然后相加，所以ω和 $ x_{i : i + h − 1} $的维度是一致的，都是h*k，所以x的维度是（n-h+1)*h*k

由于句子序列长度为n，而卷积核的高度为h，所以总共滑窗n−h+1次。所以卷积汇总结果为$c = [ c _1 , c_2 , . . . , c _{n − h + 1} ] $

### 池化

这里的池化一般采用全局最大池化