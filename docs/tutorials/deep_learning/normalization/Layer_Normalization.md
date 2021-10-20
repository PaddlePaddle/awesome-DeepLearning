# 层归一化详解

## 综述

神经网络的学习过程本质上是在学习数据的分布，如果没有进行归一化的处理，那么每一批次的训练数据的分布是不一样的，从大的方向上来看，神经网络则需要在这多个分布当中找到平衡点，从小的方向上来看 ，由于每层的网络输入数据分布在不断地变化 ，那么会导致每层网络都在找平衡点，显然网络就变得难以收敛 。当然我们可以对输入数据进行归一化处理（例如对输入图像除以255），但这也仅能保证输入层的数据分布是一样的，并不能保证每层网络输入数据分布是一样的，所以在网络的中间我们也是需要加入归一化的处理。

**归一化定义：数据标准化(Normalization)，也称为归一化，归一化就是将需要处理的数据在通过某种算法经过处理后，将其限定在需要的一定的范围内。**

## 提出背景

一般的批归一化（Batch Normalization，BN）算法对mini-batch数据集过分依赖，无法应用到在线学习任务中（此时mini-batch数据集包含的样例个数为1），在递归神经网络（Recurrent neural network，RNN）中BN的效果也不明显 ，RNN多用于自然语言处理任务，网络在不同训练周期内输入的句子，句子长度往往不同，在RNN中应用BN时，在不同时间周期使用mini-batch数据集的大小都需要不同，计算复杂，而且如果一个测试句子比训练集中的任何一个句子都长，在测试阶段RNN神经网络预测性能会出现严重偏差。如果更改为使用层归一化，就可以有效的避免这个问题。

## 概念及算法

**层归一化**：通过计算在一个训练样本上某一层所有的神经元的均值和方差来对神经元进行归一化。



$$
\mu\leftarrow\frac{1}{H}\sum_{i=1}^{H}x_i
$$

$$
\sigma\leftarrow\sqrt{\frac{1}{H}\sum_{i=1}^{H}(x_i-\mu_B)^2+\epsilon}
$$

$$
y=f(\frac{g}{\sigma}(x-\mu)+b)
$$

相关参数含义：

- `x` : 该层神经元的向量表示
- `H` : 层中隐藏神经元个数
- `ϵ` : 添加较小的值到方差中以防止除零
- `g`: 再缩放参数（可训练），新数据以$g^2$为方差
- `b`: 再平移参数（可训练），新数据以b为偏差
- `f`：激活函数

## 算法作用

1. 加快网络的训练收敛速度
   在深度神经网络中，如果每层的数据分布都不一样，将会导致网络非常难以收敛和训练（如综述所说难以在多种数据分布中找到平衡点），而每层数据的分布都相同的情况，训练时的收敛速度将会大幅度提升。
2. 控制梯度爆炸和防止梯度消失
   我们常用的梯度传递的方式是由深层神经元往浅层传播，如果用$f_{i}^\prime$和$O_i^\prime$分别表示第$i$层对应的激活层导数和输出导数，那么对于$H$层的神经网络，第一层的导数$F_1^\prime=\prod_{i=1}^{H}f_i^\prime*O_i^\prime$，那么对于$f_i^\prime*O_i^\prime$恒大于1的情况，如$f_i^\prime*O_i^\prime\equiv2$的情况，使得结果指数上升，发生梯度爆炸，对于$f_i^\prime*O_i^\prime$恒小于1，如$f_i^\prime*O_i^\prime\equiv0.25$导致结果指数下降，发生梯度消失的现象，底层神经元梯度几乎为0。采用归一化算法后，可以使得$f_i^\prime*O_i^\prime$的结果不会太大也不会太小，有利于控制梯度的传播。

## **paddle中的API** 

`paddle.nn.LayerNorm(normalized_shape, epsilon=1e-05, weight_attr=None, bias_attr=None, name=None);`

该接口用于构建 `LayerNorm` 类的一个可调用对象，具体参数详情参考[LayerNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/LayerNorm_cn.html#cn-api-nn-layernorm)。

核心参数的含义：

- `normalized_shape (int|list|tuple)` - 期望对哪些维度进行变换。如果是一个整数，会将最后一个维度进行规范化。
- `epsilon` (float, 可选) - 对应$\epsilon$-为了数值稳定加在分母上的值。默认值：1e-05

## **应用实例**：

```python
import paddle
import numpy as np

np.random.seed(123)
x_data = np.random.random(size=(2, 2, 2, 3)).astype('float32')
x = paddle.to_tensor(x_data)
layer_norm = paddle.nn.LayerNorm(x_data.shape[1:])
layer_norm_out = layer_norm(x)

print(layer_norm_out)

# input
# Tensor(shape=[2, 2, 2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[0.69646919, 0.28613934, 0.22685145],
#           [0.55131477, 0.71946895, 0.42310646]],

#          [[0.98076421, 0.68482971, 0.48093191],
#           [0.39211753, 0.34317800, 0.72904968]]],


#         [[[0.43857226, 0.05967790, 0.39804426],
#           [0.73799539, 0.18249173, 0.17545176]],

#          [[0.53155136, 0.53182757, 0.63440096],
#           [0.84943181, 0.72445530, 0.61102349]]]])

# output:
# Tensor(shape=[2, 2, 2, 3], dtype=float32, place=CPUPlace, stop_gradient=True,
#        [[[[ 0.71878898, -1.20117974, -1.47859287],
#           [ 0.03959895,  0.82640684, -0.56029880]],

#          [[ 2.04902983,  0.66432685, -0.28972855],
#           [-0.70529866, -0.93429095,  0.87123591]]],


#         [[[-0.21512909, -1.81323946, -0.38606915],
#           [ 1.04778552, -1.29523218, -1.32492554]],

#          [[ 0.17704056,  0.17820556,  0.61084229],
#           [ 1.51780486,  0.99067575,  0.51224011]]]])
```

***说明：***

对于一般的图片训练集格式为$(N,C,H,W)$的数据，在LN变换中，我们对后三个维度进行归一化。因此实例的输入shape就是后三维`x_data.shape[1:]`。也就是我们固定了以每张图片为单位，对每张图片的所有通道的像素值统一进行了`Z-score`归一化。

## 应用场景

层归一化在递归神经网络RNN中的效果是受益最大的，它的表现优于批归一化，特别是在动态长序列和小批量的任务当中 。例如在论文[Layer Normalization](https://xueshu.baidu.com/usercenter/paper/show?paperid=bce50fa3f4f88216264baf4ff6c26f5d&site=xueshu_se)所提到的以下任务当中：

1. 图像与语言的顺序嵌入（Order embedding of images and language）
2. 教机器阅读和理解（Teaching machines to read and comprehend）
3. Skip-thought向量（Skip-thought vectors）
4. 使用DRAW对二值化的MNIST进行建模（Modeling binarized MNIST using DRAW）
5. 手写序列生成（Handwriting sequence generation）
6. 排列不变MNIST（Permutation invariant MNIST）

但是，研究表明，由于在卷积神经网络中，LN会破坏卷积所学习到的特征，致使模型无法收敛，而对于BN算法，基于不同数据的情况，同一特征归一化得到的数据更不容易损失信息，所以在LN和BN都可以应用的场景，BN的表现通常要更好。

## 参考文献

> [1]https://blog.csdn.net/u013289254/article/details/99690730
>
> [2]刘建伟,赵会丹,罗雄麟,许鋆.深度学习批归一化及其相关算法研究进展[J].自动化学报,2020,46(06):1090-1120.
>
> [3] Ba J L ,  Kiros J R ,  Hinton G E . Layer Normalization[J].  2016.
>
> [4]https://blog.csdn.net/pipisorry/article/details/95906888
>
> [5]王岩. 深度神经网络的归一化技术研究[D].南京邮电大学,2019.
>
> [6]https://zhuanlan.zhihu.com/p/75603087