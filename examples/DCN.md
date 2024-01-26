# 可变形卷积-Deformable Convolution Net（DCN）【v1+v2】

## DCN v1
### **背景**
如何有效地对几何图形的变化进行建模一直是一个挑战, 大体上有两种处理方法：
1. 构建一个包含各种变化的数据集，其本质是数据扩增
1. 使用具有形变不变性的特征和算法（如SIFT）。

这两种方法都有很大的局限性：第一种方法因为样本的局限性显然模型的泛化能力比较低，无法泛化到一般场景中；在第二种方法中，几何形变被假设是固定和已知的，这是一种先验信息，用这些已知的形变去处理未知的形变是不合理的；手工设计的特征或算法无法应对过度复杂的形变，即使该形变是已知的。近年来，CNNs在计算机视觉领域取得了飞速的发展和进步，在图像分类、语义分割、目标检测领域都有很好的应用。但鉴于CNNs固定的几何结构，导致对几何形变的建模受到限制。由此提出了两个新模块来提升CNNs的形变建模能力，称为 "deformable convolution" 和 "deformable ROI pooling" ,这两个方法都是基于在模块中增加额外偏移量的空间采样位置和从目标任务中学习到偏移量且不需要额外的监督。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/17d7d2b4f2f84cb4bd1a83032a059e0b705227cbdd9d42fb9a3579272e42e154" width="500" hegiht="" ></center>
<center><br>图1：标准卷积和可变形卷积的对比</br></center>
<br></br>

### **可变形卷积**
可变形卷积顾名思义就是卷积的位置是可变形的，并非在传统的N×N的网格上做卷积，这样的好处就是更准确地提取到我们想要的特征（传统的卷积仅仅只能提取到矩形框的特征）可变卷积实际上是在每一个卷积采样点加上了一个偏移量，如下图所示：
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f418f35bff4d4ab8962044dd1f99e08e3ea61b47bcb64908ad8abfed39ff0bd1" width="500" hegiht="" ></center>
<center><br>图2：可变形卷积</br></center>
<br></br>

图2(a)所示的正常卷积规律的采样 9 个点 (绿点)， (b) (c) (d)为可变形卷积, 在正常的采样坐标上加上一个位移量 (蓝色箭头)，其中(c) (d)作为(b)的特殊情况, 展示了可变形卷积可以作为尺度变换, 比例变换和旋转变换等特殊情况。
我们先看普通的卷积, 以3x3卷积为例：对于每个输出 $\mathrm{y}(\mathbf{p}_{0})$, 都要从 $\mathrm{x}(\mathbf{p}_{0})$上采样9个位置, 这9个位置都在中心位置 x(p0)向四周扩散, $(-1,-1)$ 代表 $\mathrm{x}(\mathrm{p} 0)$ 的左上角, $(1,1)$ 代表 $\mathrm{x}(\mathrm{p} 0)$ 的右下角。
$$
\mathcal{R}=\{(-1,-1),(-1,0), \ldots,(0,1),(1,1)\}
$$
所以传统的卷积输出就是 (其中 $P_{n}$ 就是网格中的n个点)
$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}\right)
$$
正如我们上面阐述的可变形卷积，他就是在传统的卷积操作上加入了一个偏移量，正是这个偏移量才让卷积变形为不规则的卷积：
$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}+\Delta \mathbf{p}_{n}\right)
$$

由于这个偏移量可以是小数，所以上式的特征值需要通过双线性插值的方法来计算：
$$
G(\mathbf{q}, \mathbf{p})=g\left(q_{x}, p_{x}\right) \cdot g\left(q_{y}, p_{y}\right)
$$

### **偏移量的计算方式**

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/e0f0dfff93c94b28bfa8a94de149cbdf56ca6ece417f495cadf6b860f8ff5fc2" width="500" hegiht="" ></center>
<center><br>图3：偏移量的计算方式</br></center>
<br></br>

对于输入的一张feature map，假设原来的卷积操作是3×3的，那么为了学习偏移量offset，我们定义另外一个3×3的卷积层（图中上面的那层），输出的维度其实就是原来feature map大小，channel数等于2N（分别表示x,y方向的偏移）

### **可变形卷积的流程**
1. 原始图片batch（大小为b\*h\*w\*c），记为U，经过一个普通卷积，卷积填充为`same`，即输出输入大小不变，对应的输出结果为（b\*h\*w\*2c)，记为V，输出的结果是指原图片batch中每个像素的偏移量（x偏移与y偏移，因此为2c）。
1. 将U中图片的像素索引值与V相加，得到偏移后的position（即在原始图片U中的坐标值），需要将position值限定为图片大小以内。position的大小为（b\*h\*w\*2c)，但position只是一个坐标值，而且还是float类型的，我们需要这些float类型的坐标值获取像素。
1. 例如取一个坐标值（a,b)，将其转换为四个整数，floor(a), ceil(a), floor(b), ceil(b)，将这四个整数进行整合，得到四对坐标（floor(a),floor(b)),  ((floor(a),ceil(b)),  ((ceil(a),floor(b)),  ((ceil(a),ceil(b))。这四对坐标每个坐标都对应U中的一个像素值，而我们需要得到(a,b)的像素值，这里采用双线性差值的方式计算（一方面得到的像素准确，另一方面可以进行反向传播）。
1. 在得到position的所有像素后，即得到了一个新图片M，将这个新图片M作为输入数据输入到别的层中，如普通卷积。


## DCN v2
### **背景**
尽管可变形卷积网络在几何变化建模方面具有优越的性能，但其空间支持远远超出了感兴趣的区域，导致**特征受到无关图像内容的影响**。于是提出了一种新的可变形的卷积神经网络DCN v2，通过增强建模能力和更强的训练策略，提高了它对相关图像区域的聚焦能力。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c1e73e08561740a5a39e261f5d59f9f649030e2e2dbb48e1939652b82d20b788" width="500" hegiht="" ></center>
<center><br>图4：标准卷积、DCNv1、DCNv2对比</br></center>
<br></br>

从图4中可以得到几个结论：
1、基于常规卷积层的深度网络对于形变目标有一定的学习能力，比如(a)中的最后一行，基本上都能覆盖对应的目标区域或者非目标区域，这主要归功于深度网络的拟合能力，这种拟合能力有点强行拟合的意思，所以才有DCN这种设计。
2、DCN v1对于形变目标的学习能力要比常规卷积强，能够获取更多有效的信息。比如(b)中的最后一行，当输出点位置在目标上时（前2张图），影响区域相比常规卷积而言更大。
3、DCN v2对于形变目标的学习能力比DCNv1更强，不仅能获取更多有效的信息，而且获取的信息更加准确，比如©中的最后一行，目标区域更加准确。因此简单来讲，DCNv1在有效信息获取方面的recall要高于常规卷积，而DCNv2不仅有较高的recall，而且有较高的precision，从而实现信息的精确提取。

### **DCN v2的改进**
DCN v2相较于DCN v1，在以下三个方面做了改进：
（1）More Deformable Conv Layers（使用更多的可变形卷积）。

（2）Modulated Deformable Modules（在DCNv1基础（添加offset）上添加每个采样点的权重）

（3）R-CNN Feature Mimicking（模拟R-CNN的feature）。

### **1 使用更多的可变形卷积**
在DCN v1中只在conv 5中使用了三个可变形卷积，在DCN v2中把conv3到conv5都换成了可变形卷积，提高算法对几何形变的建模能力。
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/73f0f5dd795b4003b675e6dda0c6efb364241d05ccc3439b8791fbfa1f3fbd7e" width="500" hegiht="" ></center>
<center><br>图5：更多的可变形卷积对比</br></center>
<br></br>

### **2 在DCNv1基础（添加offset）上添加每个采样点的权重**
DCN $\mathrm{v}$ 1中的卷积是添加了一个offset $\Delta P_{n}$ :
$$
\mathbf{y}\left(\mathbf{p}_{0}\right)=\sum_{\mathbf{p}_{n} \in \mathcal{R}} \mathbf{w}\left(\mathbf{p}_{n}\right) \cdot \mathbf{x}\left(\mathbf{p}_{0}+\mathbf{p}_{n}+\Delta \mathbf{p}_{n}\right)
$$
为了解决引入了一些无关区域的问题, 在DCN v2中不只添加每一个采样点的偏移，还添加了一个权重系数 $\Delta m_{k}$, 来区分我们引入的区域是否为我们感兴趣的区域, 假如这个采样点的区域我们不感兴趣, 则把权重学习置为0即可:
$$
y(p)=\sum_{k=1}^{K} w_{k} \cdot x\left(p+p_{k}+\Delta p_{k}\right) \cdot \Delta m_{k}
$$
位置赋予权重, 这两方面保证了有效信息的准确提取。
### **3 模拟R-CNN的feature**
<center><img src="https://ai-studio-static-online.cdn.bcebos.com/73f0f5dd795b4003b675e6dda0c6efb364241d05ccc3439b8791fbfa1f3fbd7e" width="600" hegiht="" ></center>
<center><br>图6：模拟R-CNN的feature</br></center>
<br></br>

左边的网络为主网络 (Faster RCNN)，右边的网络为子网络（RCNN) 。实现上是用主网络训练过程中得到的Rol去裁剪原图，然后将裁剪到的图resize到224\times224大小作为子网络的输入，这部分最后提取的特征和主网络输出的1024维特征作为feature mimicking loss的输入，用来约束这2个特征的差异（通过一个余弦相似度计算，如下式所示)， 同时子网络通过一个分类损失进行监督学习，因为并不需要回归坐标, 所以没有回归损失。在inference阶段仅有主网络部分，因此这个操作不会在inference阶段增加计算成本。
$$
L_{\text {mimic }}=\sum_{b \in \Omega}\left[1-\cos \left(f_{\mathrm{RCNN}}(b), f_{\mathrm{FRCNN}}(b)\right)\right]
$$
简单来说，因为RCNN这个子网络的输入就是Rol在原输入图像上裁剪出来的图像，因此不存在Rol 以外区域信息的干扰，这就使得RCNN这个网络训练得到的分类结果更加可靠，以此通过一个损失函数监督主网络Faster RCNN的分类支路训练就能够使网络提取到更多Rol内部特征, 而不是自己引入的外部特征。

loss由三部分组成: mimic loss + R-CNN classification loss + Faster-RCNN loss.

* DCN v1论文https://arxiv.org/pdf/1703.06211.pdf
* DCN v2论文https://arxiv.org/pdf/1811.11168.pdf
