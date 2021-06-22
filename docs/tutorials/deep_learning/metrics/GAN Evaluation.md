# GAN评价指标

生成器G训练好后，我们需要评价生成图片的质量好坏，主要分为主观评价和客观评价，接下来分别介绍这两类方法：

## 主观评价

 人眼去观察生成的样本是否与真实样本相似。但是主观评价会存在以下问题：

* 生成图片数量较大时，观察一小部分图片可能无法代表所有图片的质量；

* 生成图片非常真实时，主观认为是一个好的GAN，但可能存在过拟合现象，人眼无法发现。

## 客观评价

因为主观评价存在一些问题，于是就有很多学者提出了GAN的客观评价方法，常用的方法：

* IS（Inception Score）
* FID（Fréchet Inception Distance）
* 其他评价方法

### IS

IS全称是Inception Score，其名字中 Inception 来源于Inception Net，因为计算这个 score 需要用到 Inception Net-V3（第三个版本的 Inception Net）。对于一个在ImageNet训练好的GAN，IS主要从以下两个方面进行评价：

* **清晰度：**把生成的图片 x 输入Inception V3模型中，将输出 1000 维(ImageNet有1000类)的向量 y ，向量每个维度的值表示图片属于某类的概率。对于一个清晰的图片，它属于某一类的概率应该非常大。

* **多样性：**如果一个模型能生成足够多样的图片，那么它生成的图片在各个类别中的分布应该是平均的，假设生成了 10000 张图片，那么最理想的情况是，1000 类中每类生成了 10 张。

IS计算公式为：

$$\begin{equation} IS(G) = exp(E_{x\sim p_g}D_{KL}(p(y|x)||\widehat{p}(y)))\end{equation}  \tag{1} $$

其中，$x\sim p$：表示从生成器生成的图片；p(y|x)：把生成的图片 x 输入到 Inception V3，得到一个 1000 维的向量 y ，即图片x属于各个类别的概率分布；

$\widehat{p}(y)$：N 个生成的图片（N 通常取 5000），每个生成图片都输入到 Inception V3 中，各自得到一个的概率分布向量，然后求这些向量的平均，得到生成的图片在所有类别上的边缘分布，具体公式如下： 

$$\begin{equation} \widehat{p}(y)=\frac{1}{N}\sum\limits_{i=1}^N p\left(y|x^\left(i\right)\right)\end{equation} \tag{2}$$

$D_{KL}$：表示对$p(y|x)$和$\widehat{p}(y)$求KL散度，KL散度公式如下：

$$\begin{equation} D_{KL}\left(P\|Q\right)=\sum\limits_{i} P\left(i\right)\log\frac{P(i)}{Q \left(i\right)}\end{equation} \tag{3}$$

IS不能反映过拟合、且不能在一个数据集上训练分类模型，用来评估另一个数据集上训练的生成模型。

### FID

FID全称是Fréchet Inception Distance，计算真实图片和生成图片的Inception特征向量之间的距离。

首先将Inception Net-V3模型的输出层替换为最后一个池化层的激活函数的输出值，把生成器生成的图片和真实图片送到模型中，得到2048个激活特征。生成图像的特征均值$\mu_g$和方差$C_g$ ，以及真实图像的均值$\mu_r$和方差$C_r$，根据均值和方差计算特征向量之间的距离，此距离值即FID：

$$FID\left(P_r,P_g\right) = ||\mu_r-\mu_g|| + T_r\left(C_r+C_g-2\left(C_rC_g\right)^{1/2}\right) \tag{4}$$

其中Tr 指的是被称为「迹」的线性代数运算（即方阵主对角线上的元素之和）。

FID方法比较鲁棒，且计算高效。 

### 其他评价方法

除了上述介绍的两种GAN客观评价方法 ，更多评价方法：

Mode Score、Modifified Inception Score、AM Score、MMD、图像、Image Quality Measures、SSIM、PSNR等
