```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```

## **Inception v3论文**

### **1. Introduction**

&emsp;&emsp;作者在介绍AlexNet后，过渡到更深层网络模型的提出。然后介绍GoogLeNet 考虑了内存和计算资源，五百万个参数，比六千万参数的 AlexNet 少12倍， VGGNet 则是AlexNet 的参数三倍多。提出了GoogLeNet 更适合于大数据的处理，尤其是内存或计算资源有限制的场合。原来Inception 架构的复杂性没有清晰的描述。本文主要提出了一些设计原理和优化思路。

### **2. General Design Principles**

&emsp;&emsp;2.1避免特征表示瓶颈，尤其是在网络的前面。前馈网络可以通过一个无环图来表示，该图定义的是从输入层到分类器或回归器的信息流动。要避免严重压缩导致的瓶颈。特征表示尺寸应该温和的减少，从输入端到输出端。特征表示的维度只是一个粗浅的信息量表示，它丢掉了一些重要的因素如相关性结构。

&emsp;&emsp;2.2高纬信息更适合在网络的局部处理。在卷积网络中逐步增加非线性激活响应可以解耦合更多的特征，那么网络就会训练的更快。

&emsp;&emsp;2.3空间聚合可以通过低纬嵌入，不会导致网络表示能力的降低。

&emsp;&emsp;2.4平衡好网络的深度和宽度。通过平衡网络每层滤波器的个数和网络的层数可以是网络达到最佳性能。增加网络的宽度和深度都会提升网络的性能，但是两者并行增加获得的性能提升是最大的。所以计算资源应该被合理的分配到网络的宽度和深度。

### **3. Factorizing Convolutions with Large Filter Size**

#### **3.1. Factorization into smaller convolutions**

&emsp;&emsp;大尺寸滤波器的卷积（如5x5,7x7）引入的计算量很大。例如一个 5x5 的卷积比一个3x3卷积滤波器多25/9=2.78倍计算量。当然5x5滤波器可以学习到更多的信息。那么我们能不能使用一个多层感知器来代替这个 5x5 卷积滤波器。受到NIN的启发，用下面的方法，如图进行改进。

![](https://ai-studio-static-online.cdn.bcebos.com/f34d649c88e34feb9147d436bcdcb9a7e67cbdd51b274858a795a98974f7170e)

&emsp;&emsp;5x5卷积看做一个小的全链接网络在5x5区域滑动，我们可以先用一个3x3的卷积滤波器卷积，然后再用一个全链接层连接这个3x3卷积输出，这个全链接层我们也可以看做一个3x3卷积层。这样我们就可以用两个3x3卷积级联起来代替一个 5x5卷积。如下图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/663245eda8434f33ac68f755683f15ab66d79ca41fe64deaa34bfd708a26bfe8)

![](https://ai-studio-static-online.cdn.bcebos.com/0040289b19f3489fb12aabf3583ab626fabf95eb5894464e9b1f974c4a1efeb5)

#### **3.2. Spatial Factorization into Asymmetric Convolutions**

&emsp;&emsp;空间上分解为非对称卷积，受之前启发，把3x3的卷积核分解为3x1+1x3来代替3x3的卷积。如下图所示，两层结构计算量减少33%。

![](https://ai-studio-static-online.cdn.bcebos.com/296f0104dc2749edbc6c7ac8aeac22ffa4934d7b1aae4d1ca6d979ee125c2182)


### **4. Utility of Auxiliary Classifiers**

&emsp;&emsp;引入了附加分类器，其目的是从而加快收敛。辅助分类器其实起着着正则化的作用。当辅助分类器使用了归一化或dropout时，主分类器效果会更好。

### **5. Efficient Grid Size Reduction**

&emsp;&emsp;池化操作降低特征图大小，使用两个并行的步长为2的模块, P 和 C。P是一个池化层，然后将两个模型的响应组合到一起来更多的降低计算量。

![](https://ai-studio-static-online.cdn.bcebos.com/0d616f6519c0425ebacd9cbcddd6e896f680faa3a817453e9097187cc119c588)


### **6. Inception-v2**

&emsp;&emsp;把7x7卷积替换为3个3x3卷积。包含3个Inception部分。第一部分是35x35x288，使用了2个3x3卷积代替了传统的5x5；第二部分减小了feature map，增多了filters，为17x17x768，使用了nx1->1xn结构；第三部分增多了filter，使用了卷积池化并行结构。网络有42层，但是计算量只有GoogLeNet的2.5倍。

![](https://ai-studio-static-online.cdn.bcebos.com/f9b25c0cb0804c3eb6bacb0908f23da18d9d3cd3402f4de1aab62f37504a0896)

