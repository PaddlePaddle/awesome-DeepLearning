# 向量距离与相似度

假设当前有两个$n$维向量$x$和$y$ (除非特别说明，本文默认依此写法表示向量)，可以通过两个向量之间的距离或者相似度来判定这两个向量的相近程度，显然两个向量之间距离越小，相似度越高；两个向量之间距离越大，相似度越低。

## 1. 常见的距离计算方式

### 1.1 闵可夫斯基距离（Minkowski Distance）

$$
Minkowski \; Distance = (\sum_{i=1}^n {|x_i - y_i|}^{p})^{\frac{1}{p}}
$$

Minkowski Distane 是对多个距离度量公式概括性的表述，当$p=1$时，Minkowski Distane 便是曼哈顿距离；当$p=2$时，Minkowski Distane 便是欧式距离；Minkowski Distane 取极限的形式便是切比雪夫距离。

### 1.2 曼哈顿距离（Manhattan Distance）

$$
Manhattan \; Distance = (\sum_{i=1}^n |x_i - y_i|)
$$

### 1.3 欧式距离/欧几里得距离（Euclidean distance）

$$
Euclidean \; Distance = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

### 1.4 切比雪夫距离（Chebyshev Distance）

$$
\underset{p \rightarrow \infty}{\text{lim}} (\sum_{i=1}^n {|x_i - y_i|}^{p})^{\frac{1}{p}} = \text{max} \; (|x_i-y_i|)
$$

### 1.5 海明距离（Hamming Distance）

 在信息论中，两个等长字符串之间的海明距离是两个字符串对应位置的不同字符的个数。假设有两个字符串分别是：$x=[x_1,x_2,...,x_n]$和$y=[y_1,y_2,...,y_n]$，则两者的距离为：

$$
Hamming \; Distance  = \sum_{i=1}^{n} {\text{II}}(x_i=y_i)
$$

其中$\text{II}$表示指示函数，两者相同为1，否则为0。

### 1.6 KL散度

给定随机变量$X$和两个概率分布$P$和$Q$，KL散度可以用来衡量两个分布之间的差异性，其公式如下：

$$
KL(P||Q)= \sum_{x \in X} p(x)log\,\frac{P(x)}{Q(x)}
$$

## 2. 常见的相似度函数

### 2.1 余弦相似度（Cosine Similarity）

$$
\begin{align}
Cosine \; Similarity = \frac{x \cdot y}{|x|\cdot |y|} = \frac{\sum_{i=1}^n x_iy_i}{\sqrt{\sum_{i=1}^n x_i^2}\sqrt{\sum_{i=1}^n y_i^2}}
\end{align}
$$

### 2.2 皮尔逊相关系数 （Pearson Correlation Coefficient）

给定两个随机变量$X$和$Y$，皮尔逊相关系数可以用来衡量两者的相关程度，公式如下:


$$
\begin{align}
\rho_{x,y} &= \frac{cov(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X \sigma_Y} \\
& = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i-\bar{Y})}{\sqrt{\sum_{i=1}^n(X_i-\bar{X})^2}\sqrt{\sum_{i=1}^n(Y_i-\bar{Y})^2}}
\end{align}
$$


其中$\mu_X$和$\mu_Y$分别表示向量$X$和$Y$的均值，$\sigma_X$和$\sigma_Y$分别表示向量$X$和$Y$的标准差。

### 2.3 Jaccard 相似系数（Jaccard Coefficient）

假设有两个集合$X$和$Y$(注意这里的两者不是向量)，则其计算公式为：

$$
Jaccard(X,Y)=\frac{X\cup Y}{X\cap Y}
$$

