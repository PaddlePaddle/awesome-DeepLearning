# GAN损失函数

在训练过程中，生成器G（Generator）的目标就是尽量生成真实的图片去欺骗判别器D（Discriminator）。而D的目标就是尽量把G生成的图片和真实的图片区分开。这样，G和D构成了一个动态的“博弈过程”。

最后博弈的结果是什么？在最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

用公式表示如下：

$$\begin{equation} \mathop{min}\limits_{G}\mathop{max}\limits_{D}V(D,G) = Ε_{x\sim p_{data}(x)} \left[\log D\left(x\right)\right]+Ε_{z\sim p_{z}(z)}\left[\log \left(1 - D\left(G\left(z\right)\right)\right)\right]\end{equation} \tag{1}$$

公式左边V(D,G)表示生成图像和真实图像的差异度，采用二分类(真、假两个类别)的交叉熵损失函数。包含minG和maxD两部分：

$\mathop{max}\limits_{D}V(D,G)$表示固定生成器G训练判别器D，通过最大化交叉熵损失V(D,G)来更新判别器D的参数。D的训练目标是正确区分真实图片x和生成图片G(z)，D的鉴别能力越强，D(x)应该越大，右边第一项更大，D(G(x))应该越小，右边第二项更大。这时V(D,G)会变大，因此式子对于D来说是求最大(maxD)。

$\mathop{min}\limits_{G}\mathop{max}\limits_{D}V(D,G)$表示固定判别器D训练生成器G，生成器要在判别器最大化真、假图片交叉熵损失V(D,G)的情况下，最小化这个交叉熵损失。此时右边只有第二项有用， G希望自己生成的图片“越接近真实越好”，能够欺骗判别器，即D(G(z))尽可能得大，这时V(D, G)会变小。因此式子对于G来说是求最小(min_G)。

* $$x\sim p_{data}(x)$$：表示真实图像；
* $z\sim p_{z}(z)$：表示高斯分布的样本，即噪声；
* D(x)代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

等式的右边其实就是将等式左边的交叉商损失公式展开，并写成概率分布的期望形式。详细的推导请参见原论文[Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)。