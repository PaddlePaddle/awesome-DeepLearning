#!/usr/bin/env python
# coding: utf-8

# # 一、深度学习基础知识
# **1、损失函数方法补充**
# 
# 
# **(1)0-1损失函数(zero-one loss)**
# 		  
# 0-1损失是指预测值和目标值不相等为1， 否则为0:
# <br></br>
# <img src="https://www.zhihu.com/equation?tex=L+%28+Y+%2C+f+%28+X+%29+%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D+%7B+l+%7D+%7B+1+%2C+Y+%5Cneq+f+%28+X+%29+%7D+%5C%5C+%7B+0+%2C+Y+%3D+f+%28+X+%29+%7D+%5Cend%7Barray%7D+%5Cright.+++%5C%5C" width="500" hegiht="">  
# </center>  
# <br></br>
#   	特点：
# 
# (1)0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用.
# 
# (2)感知机就是用的这种损失函数。但是相等这个条件太过严格，因此可以放宽条件，即满足
# <img src="https://www.zhihu.com/equation?tex=%7CY+-+f%28x%29%7C+%3C+T" width="100" hegiht="">  时认为相等.
# 
# 
# **(2)绝对值损失函数**
# 
# 绝对值损失函数是计算预测值与目标值的差的绝对值：
# <br></br>
# <img src="https://www.zhihu.com/equation?tex=L%28Y%2C+f%28x%29%29+%3D+%7CY+-+f%28x%29%7C++%5C%5C" width="500" hegiht="">  
# </center>  
# <br></br>
# 
# **(3)负log对数损失函数**
# 
# 该 OP 对输入的预测结果和目标标签进行计算，返回负对数损失值。
# <br></br>
# <img src="https://ai-studio-static-online.cdn.bcebos.com/75c8a37a00484043979532e4af35afecb0c0f1762e674c07929b8b7ab94b89b7" width="500" hegiht="">  
# </center>  
# <br></br>
# 特点：
# 
# (1) log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。
# 
# (2)健壮性不强，相比于hinge loss对噪声更敏感。
# 
# (3)逻辑回归的损失函数就是log对数损失函数。
# 
# ```python
# import paddle
# import paddle.nn.functional as F
# 
# label = paddle.randn((10,1))
# prob = paddle.randn((10,1))
# cost = F.log_loss(input=prob, label=label)
# 
# ```
# 
# **(4)指数损失函数（exponential loss）**
# 
# 指数损失函数的标准形式如下：
# <br></br>
# <img src="https://www.zhihu.com/equation?tex=L%28Y%7Cf%28X%29%29+%3D+exp%5B-yf%28x%29%5D++%5C%5C" width="500" hegiht="">  
# </center>  
# <br></br>
# 特点：
# 
# (1)对离群点、噪声非常敏感。经常用在AdaBoost算法中。
# 
# 
# **(5)Hinge 损失函数**
# 
# Hinge损失函数标准形式如下：
# <br></br>
# <img src="https://www.zhihu.com/equation?tex=L%28y%2C+f%28x%29%29+%3D+max%280%2C+1-yf%28x%29%29+++%5C%5C" width="500" hegiht="">  
# </center>  
# <br></br>	
# 特点：
# 
# (1)hinge损失函数表示如果被分类正确，损失为0，否则损失就为<img src="https://www.zhihu.com/equation?tex=1-yf%28x%29" width="70" hegiht="">  。SVM就是使用这个损失函数。
# 
# (2)一般的f(x) 是预测值，在-1到1之间， y是目标值(-1或1)。其含义是，f(x) 的值在-1和+1之间就可以了，并不鼓励 |f(x)|>1 ，即并不鼓励分类器过度自信，让某个正确分类的样本距离分割线超过1并不会有任何奖励，从而使分类器可以更专注于整体的误差。
# 
# (3) 健壮性相对较高，对异常点、噪声不敏感，但它没太好的概率解释。
# ```python
# def update_weights_Hinge(m1, m2, b, X1, X2, Y, learning_rate):
#     m1_deriv = 0
#     m2_deriv = 0
#     b_deriv = 0
#     N = len(X1)
#     for i in range(N):
#         # 计算偏导数
#         if Y[i]*(m1*X1[i] + m2*X2[i] + b) <= 1:
#             m1_deriv += -X1[i] * Y[i]
#             m2_deriv += -X2[i] * Y[i]
#             b_deriv += -Y[i]
#         # 否则偏导数为0
#     # 我们减去它，因为导数指向最陡的上升方向
#     m1 -= (m1_deriv / float(N)) * learning_rate
#     m2 -= (m2_deriv / float(N)) * learning_rate
#     b -= (b_deriv / float(N)) * learning_rate
# return m1, m2, b
# ```
# 
# **(6)感知损失(perceptron loss)函数**
# 
# 感知损失函数的标准形式如下：
# <br></br>
# <img src="https://www.zhihu.com/equation?tex=L%28y%2C+f%28x%29%29+%3D+max%280%2C+-f%28x%29%29++%5C%5C" width="500" hegiht="">  
# </center>  
# <br></br>	
# 特点：
# 
# (1)是Hinge损失函数的一个变种，Hinge loss对判定边界附近的点(正确端)惩罚力度很高。而perceptron loss只要样本的判定类别正确的话，它就满意，不管其判定边界的距离。它比Hinge loss简单，因为不是max-margin boundary，所以模型的泛化能力没 hinge loss强。
# 

# **3、池化方法补充**
# 
# 
# **(1)一般池化(General Pooling)**
# 		  
# <br></br>
# <img src="https://ai-studio-static-online.cdn.bcebos.com/28b614d1b58249e1b09b849fc63df3c957c80cc34acc4045b77c585285fb2262" width="400" hegiht="">  
# </center>  
# <br></br>
#   	
#    池化作用于图像中不重合的区域（与卷积操作不同），定义池化窗口的大小为sizeX，即图中红色正方形的边长，定义两个相邻池化窗口的水平位移 / 竖直位移为stride。一般池化由于每一池化窗口都是不重复的，所以sizeX=stride。
#    
#  **(2)随机池化(Stochastic Pooling)**
#  
#  <br></br>
# <img src=" https://ai-studio-static-online.cdn.bcebos.com/65d71436a1b542f6b9e5bae420b98e45533157499e0a46bb8485dc924866420e" width="400" hegiht=""> 
# </center>  
# <br></br>
# 
# Stochastic pooling是一种简单有效的正则化CNN的方法，能够降低max pooling的过拟合现象，提高泛化能力。对于pooling层的输入，根据输入的多项式分布随机选择一个值作为输出。训练阶段和测试阶段的操作略有不同。
# 
# 训练阶段：
# 
# 1）前向传播：先将池化窗口中的元素全部除以它们的和，得到概率矩阵；再按照概率随机选中的方格的值，作为该区域池化后的值。
# 
# 2）反向传播：求导时，只需保留前向传播中已经被选中节点的位置的值，其它值都为0，类似max-pooling的反向传播。
# 
# 测试阶段：
# 
# 在测试时也使用Stochastic Pooling会对预测值引入噪音，降低性能。取而代之的是使用概率矩阵加权平均。比使用Average Pooling表现要好一些。在平均意义上，与Average Pooling近似，在局部意义上，服从Max Pooling准则。
# 
# 求值示例：[https://www.cnblogs.com/tornadomeet/archive/2013/11/19/3432093.html](http://)
# 
#  **(3)重叠池化(Overlapping Pooling)**
#  
#  重叠池化，即相邻池化窗口之间会有重叠区域。如果定义池化窗口的大小为sizeX，定义两个相邻池化窗口的水平位移 / 竖直位移为stride，此时sizeX>stride。
# 
# Alexnet中提出和使用，不仅可以提升预测精度，同时一定程度上可以减缓过拟合。相比于正常池化（步长s=2，窗口x=2），重叠池化(步长s=2，窗口x=3) 可以减少top-1, top-5的错误率分别为0.4% 和0.3%。
# 
#  **(4)混合池化(Mixed Pooling)**
#  <br></br>
# <img src="https://img-blog.csdnimg.cn/20190413174230942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI1NjE0Nzcz,size_16,color_FFFFFF,t_70" width="500" hegiht=""> 
# </center>  
# <br></br>

# **4、数据增强方法修改及补充**
# 
# ### 图像
# **(1)对比度拉升**
# 
# 采用了线性函数对图像的灰度值进行变换
# 
# **(2)Gamma校正**
# 
# 采用了非线性函数（指数函数）对图像的灰度值进行变换
# 
# 这两种方式的实质是对感兴趣的图像区域进行展宽，对不感兴趣的背景区域进行压缩，从而达到图像增强的效果
# 
# **(3)直方图均衡化**
# 
# 将原始图像的直方图通过积分概率密度函数转化为概率密度为1（理想情况）的图像，从而达到提高对比度的作用。直方图均衡化的实质也是一种特定区域的展宽，但是会导致整个图像向亮的区域变换。当原始图像给定时，对应的直方图均衡化的效果也相应的确定了。
# 
# **(4)直方图规定化**
# 
# 针对直方图均衡化的存在的一些问题，将原始图像的直方图转化为规定的直方图的形式。一般目标图像的直方图的确定需要参考原始图像的直方图，并利用多高斯函数得到。
# 
# **(5)同态滤波器**
# 
# 图像的灰度图像f(x,y)可以看做为入射光分量和反射光分量两部分组成：f(x,y)=i(x,y)r(x,y).入射光比较的均匀，随着空间位置变化比较小，占据低频分量段。反射光由于物体性质和结构特点不同从而反射强弱很不相同的光，随着空间位置的变化比较的剧烈。占据着高频分量。基于图像是由光照谱和反射谱结合而成的原理设计的。
# 
# ### 其他
# **色彩抖动**
# 
# 在实际工程中为了消除图像在不同背景中存在的差异性，通常会做一些色彩抖动操作，扩充数据集合。色彩抖动主要是在图像的颜色方面做增强，主要调整的是图像的亮度，饱和度和对比度。工程中不是任何数据集都适用，通常如果不同背景的图像较多，加入色彩抖动操作会有很好的提升。
# 
# **几何变换类**
# 
# 几何变换类即对图像进行几何变换，包括翻转，旋转，裁剪，变形，缩放等各类操作。
# 
# **颜色变换**
# 
# 包括噪声、模糊、颜色变换、擦除、填充等等。
# 
# **GAN**
# 
# 通过生成对抗网络生成同类型的数据。比如生成汽车、人脸图片。通过图像风格迁移的手段，还可以生成同一物体再不同环境下的图片。

# **5、图像分类方法综述**
# 
# ### 传统方法
# 
# **SVM支持向量机**
# 
# 支持向量机（SVM）是一种强大而灵活的有监督机器学习算法是多维空间中超平面上不同类的表示。目标是分裂将数据集分成类，寻找最大边缘超平面。它建立了一个超平面或一组高维空间中的超平面和两类之间的良好分离是通过到任何类中最近的训练数据点距离最大的超平面。真正的力量该算法的性能取决于所使用的核函数。
# 
# **KNN**
# 
# K-近邻（K-NN）是一种非参数的惰性学习算法，用于分类和分类回归。该算法简单地依赖于特征向量和分类器之间的距离通过在k-最近的例子中找到最常见的类来获得未知的数据点。
# 
# **BP 神经网络**
# 
# BP（Back Propagation）网络是1986年由Rumelhart和McCelland为首的科学家小组提出，是一种按误差逆传播算法训练的多层前馈网络。它的学习规则是使用最速下降法，通过反向传播来不断调整网络的权值和阈值，使网络的误差平方和最小。BP神经网络模型拓扑结构包括输入层（input）、隐层(hide layer)和输出层(output layer)。
# 
# ### 深度学习方法
# 
# **卷积神经网络**
# 
# 卷积神经网络（CNN，或ConvNet）是一种多层神经网络，旨在通过最少的预处理直接从像素图像中识别视觉模式。这是一个特殊的人工神经网络结构。它包括两个重要的元素，即卷积层和池化层。
# 
# 
# **迁移学习**
# 
# 转移学习是一种机器学习技术，首先在机器上训练神经网络模型与正在解决的问题类似的问题，并且存储在解决过程中获得的知识解决一个问题并将其应用于不同但相关的问题。
# 

# # MNIST手写数字识别

# ### （一）准备数据
# 
# (1)数据集介绍
# 
# MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/fc73217ae57f451a89badc801a903bb742e42eabd9434ecc8089efe19a66c076)
# 
# (2)transform函数是定义了一个归一化标准化的标准
# 
# (3)train_dataset和test_dataset
# 
# paddle.vision.datasets.MNIST()中的mode='train'和mode='test'分别用于获取mnist训练集和测试集
# 
# transform=transform参数则为归一化标准

# ### （二）搭建网络
# -----
# 本是一共使用两个网络inceptionV1和Resnet

# ### inceptionV1
# 
# Inception v1的网络，将1x1，3x3，5x5的conv和3x3的pooling，堆叠在一起，一方面增加了网络的width，另一方面增加了网络对尺度的适应性；
# 
# 由图： 	
# <br></br>
# <img src="https://ai-studio-static-online.cdn.bcebos.com/b2bfa4201a1b4e77bb343339785378af2523f7155b144177925745958db09712" width="400" hegiht="">  
# </center>  
# <br></br>
# 
# Inception v1的亮点：
# 
# 1.卷积层共有的一个功能，可以实现通道方向的降维和增维，至于是降还是增，取决于卷积层的通道数（滤波器个数），在Inception v1中1*1卷积用于降维，减少weights大小和feature map维度。
# 
# 2.1*1卷积特有的功能，由于1*1卷积只有一个参数，相当于对原始feature map做了一个scale，并且这个scale还是训练学出来的，无疑会对识别精度有提升。
# 
# 3.增加了网络的深度
# 
# 4.增加了网络的宽度
# 
# 5.同时使用了1*1，3*3，5*5的卷积，增加了网络对尺度的适应性
# 

# ### Resnet
# ResNet网络是参考了VGG19网络，在其基础上进行了修改，并通过短路机制加入了残差单元，如图5所示。变化主要体现在ResNet直接使用stride=2的卷积做下采样，并且用global average pool层替换了全连接层。ResNet的一个重要设计原则是：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度。从图5中可以看到，ResNet相比普通网络每两层间增加了短路机制，这就形成了残差学习，其中虚线表示feature map数量发生了改变。图5展示的34-layer的ResNet，还可以构建更深的网络如表1所示。从表中可以看到，对于18-layer和34-layer的ResNet，其进行的两层间的残差学习，当网络更深时，其进行的是三层间的残差学习，三层卷积核分别是1x1，3x3和1x1，一个值得注意的是隐含层的feature map数量是比较小的，并且是输出feature map数量的1/4。
# <br></br>
# <img src="https://pic2.zhimg.com/80/v2-7cb9c03871ab1faa7ca23199ac403bd9_720w.jpg" width="400" hegiht="">  
# </center>  
# <br></br>
# 
# 而我们这次使用的是50层的resnet。

# ### 优化器和损失函数
# 
# 优化器使用Adam
# 损失函数为交叉熵损失函数
# 
# ------
# **Adam**
# 
# Adam 是一种可以替代传统随机梯度下降过程的一阶优化算法，它能基于训练数据迭代地更新神经网络权重。
# 
# 2014年12月，Kingma和Lei Ba两位学者提出了Adam优化器，结合AdaGrad和RMSProp两种优化算法的优点。对梯度的一阶矩估计（First Moment Estimation，即梯度的均值）和二阶矩估计（Second Moment Estimation，即梯度的未中心化的方差）进行综合考虑，计算出更新步长。
# 
# 主要包含以下几个显著的优点：
# 
# 1. 实现简单，计算高效，对内存需求少
# 
# 2. 参数的更新不受梯度的伸缩变换影响
# 
# 3. 超参数具有很好的解释性，且通常无需调整或仅需很少的微调
# 
# 4. 更新的步长能够被限制在大致的范围内（初始学习率）
# 
# 5. 能自然地实现步长退火过程（自动调整学习率）
# 
# 6. 很适合应用于大规模的数据及参数的场景
# 
# 7. 适用于不稳定目标函数
# 
# 8. 适用于梯度稀疏或梯度存在很大噪声的问题
# 
# 综合Adam在很多情况下算作默认工作性能比较优秀的优化器。
# 
# ------
# **交叉熵损失函数**
# 
# 对数损失Log Loss ，也被称为交叉熵损失Cross-entropy Loss，是定义在概率分布的基础上的。它通常用于多项式(multinomia)logistic regression 和神经网络，还有在期望极大化算法(expectation-maximization)的一些变体中。
# 
# 对数损失用来度量分类器的预测输出的概率分布(predict_proba)和真实分布的差异，而不是去比较离散的类标签是否相同。

# In[1]:


from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter

log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

class InceptionA(paddle.nn.Layer):  #作为网络一层
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch3x3_1=paddle.nn.Conv2D(in_channels,16,kernel_size=1) #第一个分支
        self.branch3x3_2=paddle.nn.Conv2D( 16,24,kernel_size=3,padding=1)
        self.branch3x3_3=paddle.nn.Conv2D(24,24,kernel_size=3,padding=1)

        self.branch5x5_1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第二个分支
        self.branch5x5_2=paddle.nn.Conv2D( 16,24,kernel_size=5,padding=2)

        self.branch1x1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第三个分支

        self.branch_pool=paddle.nn.Conv2D(in_channels,24,kernel_size= 1) #第四个分支

    def forward(self,x):
        #分支1处理过程
        branch3x3= self.branch3x3_1(x)
        branch3x3= self.branch3x3_2(branch3x3)
        branch3x3= self.branch3x3_3(branch3x3)
        #分支2处理过程
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        #分支3处理过程
        branch1x1=self.branch1x1(x)
        #分支4处理过程
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding= 1)
        branch_pool=self.branch_pool(branch_pool)
        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]     #将4个分支的输出拼接起来
        return fluid.layers.concat(outputs,axis=1) #横着拼接， 共有24+24+16+24=88个通道

class Net(paddle.nn.Layer):        #卷积，池化，inception，卷积，池化，inception，全连接
    def __init__(self):
        super(Net,self).__init__()
        #定义两个卷积层
        self.conv1=paddle.nn.Conv2D(1,10,kernel_size=5)
        self.conv2=paddle.nn.Conv2D(88,20,kernel_size=5)
        #Inception模块的输出均为88通道
        self.incep1=InceptionA(in_channels=10 )
        self.incep2=InceptionA(in_channels=20)
        self.mp=paddle.nn.MaxPool2D(2)
        self.fc=paddle.nn.Linear(1408,10) #5*5* 88 =2200，图像高*宽*通道数
    def forward(self,x):
        x=F.relu(self.mp(self.conv1(x)))# 卷积池化，relu  输出x为图像尺寸14*14*10
        x =self.incep1(x)               #图像尺寸14*14*88

        x =F.relu(self.mp(self.conv2(x)))# 卷积池化，relu  输出x为图像尺寸5*5*20
        x = self.incep2(x)              #图像尺寸5*5*88

        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.fc(x)
        return x
model = paddle.Model(Net())   # 封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()) # adam优化器

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,epochs=2,batch_size=64,verbose=1)
#评估
model.evaluate(test_dataset, batch_size=64, verbose=1)


# In[3]:


from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter

log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

class InceptionA(paddle.nn.Layer):  #作为网络一层
    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch3x3_1=paddle.nn.Conv2D(in_channels,16,kernel_size=1) #第一个分支
        self.branch3x3_2=paddle.nn.Conv2D( 16,24,kernel_size=3,padding=1)
        self.branch3x3_3=paddle.nn.Conv2D(24,24,kernel_size=3,padding=1)

        self.branch5x5_1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第二个分支
        self.branch5x5_2=paddle.nn.Conv2D( 16,24,kernel_size=5,padding=2)

        self.branch1x1=paddle.nn.Conv2D(in_channels, 16,kernel_size=1) #第三个分支

        self.branch_pool=paddle.nn.Conv2D(in_channels,24,kernel_size= 1) #第四个分支

    def forward(self,x):
        #分支1处理过程
        branch3x3= self.branch3x3_1(x)
        branch3x3= self.branch3x3_2(branch3x3)
        branch3x3= self.branch3x3_3(branch3x3)
        #分支2处理过程
        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)
        #分支3处理过程
        branch1x1=self.branch1x1(x)
        #分支4处理过程
        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding= 1)
        branch_pool=self.branch_pool(branch_pool)
        outputs=[branch1x1,branch5x5,branch3x3,branch_pool]     #将4个分支的输出拼接起来
        return fluid.layers.concat(outputs,axis=1) #横着拼接， 共有24+24+16+24=88个通道

class Net(paddle.nn.Layer):        #卷积，池化，inception，卷积，池化，inception，全连接
    def __init__(self):
        super(Net,self).__init__()
        #定义两个卷积层
        self.conv1=paddle.nn.Conv2D(1,10,kernel_size=5)
        self.conv2=paddle.nn.Conv2D(88,20,kernel_size=5)
        #Inception模块的输出均为88通道
        self.incep1=InceptionA(in_channels=10 )
        self.incep2=InceptionA(in_channels=20)
        self.mp=paddle.nn.MaxPool2D(2)
        self.fc=paddle.nn.Linear(1408,10) #5*5* 88 =2200，图像高*宽*通道数
    def forward(self,x):
        x=F.relu(self.mp(self.conv1(x)))# 卷积池化，relu  输出x为图像尺寸14*14*10
        x =self.incep1(x)               #图像尺寸14*14*88

        x =F.relu(self.mp(self.conv2(x)))# 卷积池化，relu  输出x为图像尺寸5*5*20
        x = self.incep2(x)              #图像尺寸5*5*88

        x = paddle.flatten(x, start_axis=1,stop_axis=-1)
        x = self.fc(x)
        return x

#训练
def train(model,Batch_size=64):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    model.train()
    iterator = 0
    epochs = 10
    total_steps = (int(50000//Batch_size)+1)*epochs
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01,decay_steps=total_steps,end_lr=0.001)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
                log_writer.add_scalar(tag='acc',step=iterator,value=acc.numpy())
                log_writer.add_scalar(tag='loss',step=iterator,value=loss.numpy())
                iterator+=200
            optim.step()
            optim.clear_grad()
        paddle.save(model.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdparams')
        paddle.save(optim.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdopt')


#测试
def test(model):
    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    model.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

#随机抽取100张图片进行测试
def random_test(model,num=100):
    select_id = random.sample(range(1, 10000), 100) #生成一百张测试图片的下标
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        label = data[1]
    predicts = model(x_data)
    #返回正确率
    acc = paddle.metric.accuracy(predicts, label)
    print("正确率为：{}".format(acc.numpy()))


if __name__ == '__main__':
    model = Net()
    train(model)
    test(model)
    random_test(model)


# In[1]:


from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock



log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

# class Net(paddle.nn.Layer):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.layer=ResNet(BottleneckBlock, 50,10).conv1= paddle.nn.Conv2D(1,64,kernel_size=7,stride=2,padding=3)
#         self.fc = paddle.nn.Linear(1000, 10)
#     #网络的前向计算过程
#     def forward(self,x):
#         x=self.layer(x)
#         print(x.shape)
#         x=self.fc(x)
#         return x

# model = paddle.Model(Net())   # 封装模型
model = resnet50 = ResNet(BottleneckBlock, 50,10)
model.conv1= paddle.nn.Conv2D(1,64,kernel_size=7,stride=2,padding=3)
# print("###:",model)
model = paddle.Model(model)   # 封装模型
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()) # adam优化器

# 配置模型
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )
# 训练模型
model.fit(train_dataset,epochs=2,batch_size=64,verbose=1)
#评估
model.evaluate(test_dataset, batch_size=64, verbose=1)


# In[2]:


from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock, BasicBlock



log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])
#归一化

#读取训练集 测试集数据
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)



def train(model,Batch_size=64):
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    model.train()
    iterator = 0
    epochs = 10
    total_steps = (int(50000//Batch_size)+1)*epochs
    lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01,decay_steps=total_steps,end_lr=0.001)
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
                log_writer.add_scalar(tag='acc',step=iterator,value=acc.numpy())
                log_writer.add_scalar(tag='loss',step=iterator,value=loss.numpy())
                iterator+=200
            optim.step()
            optim.clear_grad()
        paddle.save(model.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdparams')
        paddle.save(optim.state_dict(),'./data/checkpoint/mnist_epoch{}'.format(epoch)+'.pdopt')


#测试
def test(model):
    # 加载测试数据集
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    model.eval()
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

#随机抽取100张图片进行测试
def random_test(model,num=100):
    select_id = random.sample(range(1, 10000), 100) #生成一百张测试图片的下标
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        label = data[1]
    predicts = model(x_data)
    #返回正确率
    acc = paddle.metric.accuracy(predicts, label)
    print("正确率为：{}".format(acc.numpy()))


if __name__ == '__main__':
    model = resnet50 = ResNet(BottleneckBlock, 50,10)
    model.conv1= paddle.nn.Conv2D(1,64,kernel_size=7,stride=2,padding=3)
    train(model)
    test(model)
    random_test(model)

