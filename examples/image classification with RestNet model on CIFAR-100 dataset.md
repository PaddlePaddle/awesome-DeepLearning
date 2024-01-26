

# image classification with RestNet model 

# 	on CIFAR-100 dataset

# 实验目的：

  了解飞浆实训平台，掌握飞浆实训平台的基本操作和使用方法。通过实操实训，巩固深度学习基础知识，入门深度学习，感受深度学习的魅力。尝试使用深度学习完成图像识别。了解使用CIFAR100 数据集，学习ResNet 网络原理，尝试实现图像分类。尝试了解和学习有关网络结构、损失函数、优化算法、训练调试与优化等操作的含义，进一步了区分和了解人工智能，机器学习与深度学习之间的联系与差别。旨在通过实际操作，去理解和感受深度学习的魅力和原理，掌握从数据处理，模型设计，到模型训练和测试的全流程，激发对于深入学习深度学习的兴趣。

# 飞浆链接：

​	https://aistudio.baidu.com/aistudio/projectdetail/2261622



## 实验步骤

### 一：理论讲解：

#### 1. 了解CIFAR100数据集：

  CIFAR100数据集有100个类。每个类有600张大小为32 × 32 32\times 3232×32的彩色图像，其中500张作为训练集，100张作为测试集。对于每一张图像，它有fine_labels和coarse_labels两个标签，分别代表图像的细粒度和粗粒度标签，对应下图中的classes和superclass。也就是说，CIFAR100数据集是层次的。CIFAR-100中的100个类别分为20个超类super-class。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）。

#### 2：了解RestNet（残差网络）

  残差在数理统计中是指实际观察值与估计值（[拟合值](https://baike.baidu.com/item/拟合值/9461734)）之间的差。在集成学习中可以通过基模型拟合残差，使得集成的模型变得更精确；在深度学习中有人利用layer去拟合残差将深度神经网络的性能提高变强。

  一般认为神经网络的每一层分别对应于提取不同层次的特征信息，有低层，中层和高层，而网络越深的时候，提取到的不同层次的信息会越多，而不同层次间的层次信息的组合也会越多。

  深度学习对于网络深度遇到的主要问题是梯度消失和梯度爆炸，传统对应的解决方案则是数据的初始化(normlized initializatiton)和（batch normlization）正则化，但是这样虽然解决了梯度的问题，深度加深了，却带来了另外的问题，就是网络性能的退化问题，深度加深了，错误率却上升了，而残差用来设计解决退化问题，其同时也解决了梯度问题，更使得网络的性能也提升了。

![https://img-blog.csdnimg.cn/img_convert/0002a4dba990156e992f7ccb5f471f11.png](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

  它对每层的输入做一个reference(X), 学习形成残差函数， 而不是学习一些没有reference(X)的函数。这种残差函数更容易优化，能使网络层数大大加深。在上图的残差块中它有二层，如下表达式，

其中σ代表非线性函数ReLU。

![146cde8bfbb42807c71fa9e8fab662ad.png](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

然后通过一个shortcut，和第2个ReLU，获得输出y。

![beb76fc035e4ce97fb77646fcc8d10df.png](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg)

当需要对输入和输出维数进行变化时(如改变通道数目)，可以在shortcut时对x做一个线性变换Ws，如下式。

![e81f6c5eee21dd8ac7fb6b8d5ac5d205.png](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg)

然而实验证明x已经足够了，不需要再搞个维度变换，除非需求是某个特定维度的输出，如是将通道数翻倍

​    将输入叠加到下层的输出上。对于一个堆积层结构（几层堆积而成）当输入为x时其学习到的特征记为H(x)，现在我们希望其可以学习到残差F(x)=H(x)-x，这样其实原始的学习特征是F(x)+x 。当残差为0时，此时堆积层仅仅做了恒等映射，至少网络性能不会下降，实际上残差不会为0，这也会使得堆积层在输入特征基础上学习到新的特征，从而拥有更好的性能。

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

#### 3：RestNet的分类：

  残差网络按照其网络层数的不同，可分为：

18是简化版

34是18翻个倍

50是34的加bottleneck版，实际没变

101是在50的基础上加厚第四层的卷积块

152是在50的基础上加厚第三层和第四层的卷积块

 

![9eb1571b85a4d0b2facfa3ad7662ce56.png](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg)

 

 

 

 

### 二：代码实现

在本地搭建好python环境，导入paddle包，导入paddle遇到问题，导入paddlepaddle成功。

使用Paddle中的残差网络完成CIFAR100图像分类

#### 1.导入数据

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image015.jpg)

#### 2.数据准备

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image017.jpg)

运行结果：（数据集下载完成）

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image018.png)![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image020.jpg)

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image022.jpg)

#### 3.模型选择

（由于训练时间太长了，故随机选取一个模型尝试）

#### ![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image023.png)

#### 4.封装模型可视化：

此处学习率是一个较为关键的参数，学习率过小会导致次数较少是难以达到最优模型，过大，会错过最优，需要反复确定参数。

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image025.jpg)

运行结果：

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image027.jpg)

 

#### 5.模型训练

考虑到时间关系，训练20轮，但训练速度还是出奇的慢！

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image029.jpg)

6.模型保存（将训练好的模型保存，后续可直接使用）

![img](file:///C:/Users/17655/AppData/Local/Temp/msohtmlclip1/01/clip_image031.jpg)

#### 实验结果：

训练20轮，花费10余个小时：

```
The loss value printed in the log is the current step, and the metric is the average value of previous steps.
Epoch 1/20
D:\Python\venv\lib\site-packages\paddle\fluid\layers\utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return (isinstance(seq, collections.Sequence) and
D:\Python\venv\lib\site-packages\paddle\nn\layer\norm.py:641: UserWarning: When training, we now always track global mean and variance.
  "When training, we now always track global mean and variance.")
step 196/196 [==============================] - loss: 4.5551 - acc: 0.0160 - 8s/step           
Eval begin...
step 40/40 [==============================] - loss: 4.4025 - acc: 0.0295 - 2s/step           
Eval samples: 10000
Epoch 2/20
step 196/196 [==============================] - loss: 4.0349 - acc: 0.0522 - 8s/step           
Eval begin...
step 40/40 [==============================] - loss: 4.0835 - acc: 0.0748 - 2s/step           
Eval samples: 10000
Epoch 3/20
step 196/196 [==============================] - loss: 4.0160 - acc: 0.0893 - 8s/step           
Eval begin...
step 40/40 [==============================] - loss: 4.0470 - acc: 0.1060 - 2s/step           
Eval samples: 10000
Epoch 4/20
step 196/196 [==============================] - loss: 3.8161 - acc: 0.1184 - 8s/step           
Eval begin...
step 40/40 [==============================] - loss: 4.1403 - acc: 0.1303 - 2s/step           
Eval samples: 10000
Epoch 5/20
step 196/196 [==============================] - loss: 3.5009 - acc: 0.1417 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.6699 - acc: 0.1587 - 2s/step           
Eval samples: 10000
Epoch 6/20
step 196/196 [==============================] - loss: 3.2808 - acc: 0.1707 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.7980 - acc: 0.1863 - 2s/step           
Eval samples: 10000
Epoch 7/20
step 196/196 [==============================] - loss: 3.2443 - acc: 0.2003 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.5357 - acc: 0.1940 - 2s/step           
Eval samples: 10000
Epoch 8/20
step 196/196 [==============================] - loss: 2.8920 - acc: 0.2287 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.6978 - acc: 0.2125 - 2s/step           
Eval samples: 10000
Epoch 9/20
step 196/196 [==============================] - loss: 2.5639 - acc: 0.2553 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.8123 - acc: 0.2416 - 2s/step           
Eval samples: 10000
Epoch 10/20
step 196/196 [==============================] - loss: 2.9998 - acc: 0.2861 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.6459 - acc: 0.1655 - 2s/step           
Eval samples: 10000
Epoch 11/20
step 196/196 [==============================] - loss: 3.0642 - acc: 0.3155 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.9759 - acc: 0.2895 - 2s/step           
Eval samples: 10000
Epoch 12/20
step 196/196 [==============================] - loss: 2.8012 - acc: 0.3406 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.5134 - acc: 0.2656 - 2s/step           
Eval samples: 10000
Epoch 13/20
step 196/196 [==============================] - loss: 2.6663 - acc: 0.3717 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.6458 - acc: 0.3145 - 2s/step           
Eval samples: 10000
Epoch 14/20
step 196/196 [==============================] - loss: 1.9645 - acc: 0.3969 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.3124 - acc: 0.3068 - 2s/step           
Eval samples: 10000
Epoch 15/20
step 196/196 [==============================] - loss: 2.4781 - acc: 0.4182 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.5437 - acc: 0.3235 - 2s/step           
Eval samples: 10000
Epoch 16/20
step 196/196 [==============================] - loss: 2.4209 - acc: 0.4397 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.7646 - acc: 0.3175 - 2s/step           
Eval samples: 10000
Epoch 17/20
step 196/196 [==============================] - loss: 2.2981 - acc: 0.4687 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.0377 - acc: 0.2958 - 2s/step           
Eval samples: 10000
Epoch 18/20
step 196/196 [==============================] - loss: 1.8527 - acc: 0.4985 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 3.1634 - acc: 0.2676 - 2s/step           
Eval samples: 10000
Epoch 19/20
step 196/196 [==============================] - loss: 1.7763 - acc: 0.4998 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.9121 - acc: 0.3280 - 2s/step           
Eval samples: 10000
Epoch 20/20
step 196/196 [==============================] - loss: 1.7708 - acc: 0.5151 - 7s/step           
Eval begin...
step 40/40 [==============================] - loss: 2.9470 - acc: 0.3469 - 2s/step           
Eval samples: 10000
Eval begin...
step 40/40 [==============================] - loss: 2.9470 - acc: 0.3469 - 2s/step           
Eval samples: 10000
{'loss': [2.9469934], 'acc': 0.3469}

Process finished with exit code 0
```



可以明显的看到。经过十五轮的训练，

loss由4.9423 下降到2.4515

训练样本的识别准确率由0.024 上升到了0.7765

测试样本的识别准确率由0.036 上升到了0.1688

可以明显的看到。到第二十轮精确度上升到了90.81%

在第十四轮时 大到0.3969%

由于我的设置的学习率较小，且训练时间较短，并没有达到很高的准确度，但若修改这两个参数，耐心等待训练完成，预计准确率能有一个较大的提升。

在我的电脑上，完成一轮训练耗费约半小时。

#### 实训总结：

  本实训任务有一定的难度和挑战性，首先是对深度学习没有很好的理论基础，基本上是一张白纸，对实验难以下手。其次是对实验有抵触情绪，起初不愿意深入学习和了解。最终百度查阅了很多学习资料，借鉴了别人的思路和方法，才勉强完成本次实验。但由于训练的时间太长，实在没办法设置更多的训练轮次，导致分类识别的准确率不高。虽说过程较为艰辛，幸好还是有所收获，对残差网络的工作原理与优势有了一定的了解，通过观看飞浆课程的其他内容，对深度学习也积攒了一定的热情。再一次感受到了机器学习，人工智能和深度学习的魅力。实验目的：

  了解飞浆实训平台，掌握飞浆实训平台的基本操作和使用方法。通过实操实训，巩固深度学习基础知识，入门深度学习，感受深度学习的魅力。尝试使用深度学习完成图像识别。了解使用CIFAR100 数据集，学习ResNet 网络原理，尝试实现图像分类。尝试了解和学习有关网络结构、损失函数、优化算法、训练调试与优化等操作的含义，进一步了区分和了解人工智能，机器学习与深度学习之间的联系与差别。旨在通过实际操作，去理解和感受深度学习的魅力和原理，掌握从数据处理，模型设计，到模型训练和测试的全流程，激发对于深入学习深度学习的兴趣。

