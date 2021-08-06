## 基于VGG网络实现CIFAR100数据集的图像分类

### 一、VGG网络理论

VGGNet是牛津大学计算机视觉组（VisualGeometry  Group）和GoogleDeepMind公司的研究员一起研发的的深度卷积神经网络。VGGNet探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3×3的小型卷积核和2×2的最大池化层，VGGNet成功地构筑了16~19层深的卷积神经网络。VGGNet相比之前state-of-the-art的网络结构，错误率大幅下降，同时VGGNet的拓展性很强，迁移到其他图片数据上的泛化性非常好。VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3×3）和最大池化尺寸（2×2）。到目前为止，VGGNet依然经常被用来提取图像特征。

原论文地址：[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

VGG网络有以下特点：

1. 结构简洁
	VGG由5层卷积层、3层全连接层、softmax输出层构成，层与层之间使用max-pooling（最大化池）分开，所有隐层的激活单元都采用ReLU函数。
2. 小卷积核和多卷积子层
	 VGG使用多个较小卷积核（3x3）的卷积层代替一个卷积核较大的卷积层，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。
3. 小池化核
	 相比AlexNet的3x3的池化核，VGG全部采用2x2的池化核。
4. 通道数多
	 VGG网络第一层的通道数为64，后面每层都进行了翻倍，最多到512个通道，通道数的增加，使得更多的信息可以被提取出来。
5. 层数更深、特征图更宽
	 由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。

VGG网络根据深度的不同分有多个版本，其网络结构如下如所示，一般常用的是VGG-16模型：

<img src="./image classification with VGG model on CIFAR-100 dataset-image/vgg1.png" alt="image-20210804151222511" style="zoom:100%;" />

224×224×3的图片输入在VGG16网络下维度的变化：

<img src="./image classification with VGG model on CIFAR-100 dataset-image/vgg2.png" alt="img" style="zoom: 100%;" />





### 二、实验设计逻辑

本实训任务主要基于paddlepaddle深度学习框架，使用图像分类中常用的VGG网络对经典的CIFAR100数据集进行图像分类。实训任务主要分为数据处理（包括数据获取与数据增强等）、模型设计（VGG）、训练配置（训练资源、优化器等）、模型训练与评估、模型推理等几部分。



### 三、数据处理

#### 1）数据集介绍

本实训任务主要使用CIFAR100数据集，CIFAR100数据集有100个类。每个类有600张大小为32×32的彩色图像，其中500张作为训练集，100张作为测试集（总训练集共50000个样本，测试集共10000个样本）。对于每一张图像，有`fine_labels`和`coarse_labels`两个标签，分别代表图像的细粒度和粗粒度标签，对应下图中的`classes`和`superclass`，即CIFAR100数据集是层次的。

<img src="./image classification with VGG model on CIFAR-100 dataset-image/dataset.png" alt="image.png" style="zoom: 100%;" />

#### 2）数据加载及数据增强

图像数据准备对神经网络与卷积神经网络模型训练有重要影响，当样本空间不够或者样本数量不足的时候会严重影响训练或者导致训练出来的模型泛化程度不够，导致识别率与准确率不高。扩增数据集、防止数据过拟合常用的方法为数据增强。数据增强也叫数据扩增，意思是在不实质性的增加数据的情况下，让有限的数据产生等价于更多数据的价值。常用的数据增强方法有：

- 使用标准化对图像进行图像增强
- 使用几何变换（平移、翻转、旋转）对图像进行数据增强
- 使用随机调整亮度对图像进行增强
- 使用随机调整对比度对图像进行增强

飞桨框架2.0全新推出高层API，是对飞桨API的进一步封装与升级，提供了更加简洁易用的API，进一步提升了飞桨的易学易用性，并增强飞桨的功能。高层API将一些常用到的数据集作为领域API，对应API所在目录为`paddle.vision.datasets`。

```python
import paddle
import paddle.vision.transforms as t

def data_process():
    # 数据增强策略
    transform_strategy = t.Compose([
        t.ColorJitter(),            	#随即调节亮度、对比度等
        t.RandomHorizontalFlip(),  # 随机水平翻转
        t.RandomVerticalFlip(),      # 随机垂直翻转
        t.ToTensor()  			 # 转化为张量
    ])

    # 加载训练数据集
    train_dataset = paddle.vision.datasets.Cifar100(
        mode='train',
        transform=transform_strategy
    )

    # 测试集采用与训练集相同的增强策略，检验模型的泛化能力
    eval_dataset = paddle.vision.datasets.Cifar100(
        mode='test',
        transform=transform_strategy
    )

    print('训练集样本数:', str(len(train_dataset)), '| 测试集样本数:', str(len(eval_dataset)))
    return train_dataset, eval_dataset
```

> [out] 训练集样本数: 50000 | 测试集样本数: 10000

#### 3）数据集可视化验证

使用matplitlib可视化数据集，验证数据集是否经过增强处理。

```python
import matplotlib.pyplot as plt
import random

# 可视化部分数据集
def visualize(dataset):
    #生成m*n的视图矩阵
    m = 3
    n = 3
    for i in range(1, m * n + 1):
        plt.subplot(n, m, i)
        rand = random.randint(0,50000)#测试数据集中随机选取个样本
        img = dataset[i][0].numpy().transpose([1, 2, 0])
        plt.imshow(img)
        plt.title(dataset[i][1]) #图片上方显示样本标签
```

> [out]
>
> <img src="./image classification with VGG model on CIFAR-100 dataset-image/visualize_dataset.png" alt="image-20210804172518542" style="zoom:100%;" />

可发现有些样本进行了转置，样本标签显示在图片上方，如标签0表示`apple`。



### 四、模型设计

本次实训采用经典的VGG16模型。

<img src="./image classification with VGG model on CIFAR-100 dataset-image/vgg16.png" alt="image-20210804164609121" style="zoom:100%;" />

paddlepaddle2.0提供了了VGG网络的高层代码，定义好网络结构之后来使用`paddle.Model`完成模型的封装，将网络结构组合成一个可快速使用高层API进行训练、评估和预测的类。代码实现：

```python
def VGG(n=16):
    #根据参数选取不同层数的VGG模型，batch_norm使用均值和方差将每个元素标准化
    if n==11:
        vgg = paddle.vision.models.vgg11(batch_norm=True)#paddlepaddle高层调用
    elif n==13:
        vgg = paddle.vision.models.vgg13(batch_norm=True)
    elif n==19:
        vgg = paddle.vision.models.vgg19(batch_norm=True)
    else:
        vgg = paddle.vision.models.vgg16(batch_norm=True)

    #高层API构建实例
    model = paddle.Model(vgg)
    #输出模型详细信息
    model.summary((-1, 3, 32, 32))
    return model
```

以下是VGG16模型的详细信息：

> [out]
>
> ```
> -------------------------------------------------------------------------------
>    Layer (type)         Input Shape          Output Shape         Param #  
> -------------------------------------------------------------------------------
>      Conv2D-1          [[1, 3, 32, 32]]        [1, 64, 32, 32]         1,792  
>    BatchNorm2D-1   [[1, 64, 32, 32]]      [1, 64, 32, 32]           256  
>       ReLU-1             [[1, 64, 32, 32]]      [1, 64, 32, 32]             0  
>      Conv2D-2          [[1, 64, 32, 32]]      [1, 64, 32, 32]         36,928  
>    BatchNorm2D-2   [[1, 64, 32, 32]]      [1, 64, 32, 32]           256  
>       ReLU-2             [[1, 64, 32, 32]]      [1, 64, 32, 32]             0  
>     MaxPool2D-1      [[1, 64, 32, 32]]      [1, 64, 16, 16]             0  
>      Conv2D-3          [[1, 64, 16, 16]]      [1, 128, 16, 16]       73,856  
>    BatchNorm2D-3   [[1, 128, 16, 16]]    [1, 128, 16, 16]         512  
>       ReLU-3             [[1, 128, 16, 16]]    [1, 128, 16, 16]           0  
>      Conv2D-4          [[1, 128, 16, 16]]    [1, 128, 16, 16]       147,584  
>    BatchNorm2D-4   [[1, 128, 16, 16]]    [1, 128, 16, 16]          512  
>       ReLU-4             [[1, 128, 16, 16]]    [1, 128, 16, 16]           0  
>     MaxPool2D-2      [[1, 128, 16, 16]]    [1, 128, 8, 8]              0  
>      Conv2D-5          [[1, 128, 8, 8]]        [1, 256, 8, 8]          295,168  
>    BatchNorm2D-5   [[1, 256, 8, 8]]        [1, 256, 8, 8]           1,024  
>       ReLU-5             [[1, 256, 8, 8]]        [1, 256, 8, 8]              0  
>      Conv2D-6          [[1, 256, 8, 8]]        [1, 256, 8, 8]          590,080  
>    BatchNorm2D-6   [[1, 256, 8, 8]]        [1, 256, 8, 8]            1,024  
>       ReLU-6             [[1, 256, 8, 8]]        [1, 256, 8, 8]              0  
>      Conv2D-7          [[1, 256, 8, 8]]        [1, 256, 8, 8]          590,080  
>    BatchNorm2D-7   [[1, 256, 8, 8]]        [1, 256, 8, 8]           1,024  
>       ReLU-7             [[1, 256, 8, 8]]        [1, 256, 8, 8]              0  
>     MaxPool2D-3      [[1, 256, 8, 8]]        [1, 256, 4, 4]              0  
>      Conv2D-8          [[1, 256, 4, 4]]        [1, 512, 4, 4]         1,180,160  
>    BatchNorm2D-8   [[1, 512, 4, 4]]        [1, 512, 4, 4]           2,048  
>       ReLU-8             [[1, 512, 4, 4]]        [1, 512, 4, 4]              0  
>      Conv2D-9          [[1, 512, 4, 4]]        [1, 512, 4, 4]          2,359,808  
>    BatchNorm2D-9   [[1, 512, 4, 4]]        [1, 512, 4, 4]            2,048  
>       ReLU-9             [[1, 512, 4, 4]]        [1, 512, 4, 4]              0  
>      Conv2D-10        [[1, 512, 4, 4]]        [1, 512, 4, 4]          2,359,808  
>   BatchNorm2D-10  [[1, 512, 4, 4]]        [1, 512, 4, 4]            2,048  
>       ReLU-10           [[1, 512, 4, 4]]        [1, 512, 4, 4]              0  
>     MaxPool2D-4     [[1, 512, 4, 4]]        [1, 512, 2, 2]               0  
>      Conv2D-11        [[1, 512, 2, 2]]        [1, 512, 2, 2]          2,359,808  
>   BatchNorm2D-11  [[1, 512, 2, 2]]        [1, 512, 2, 2]            2,048  
>       ReLU-11           [[1, 512, 2, 2]]        [1, 512, 2, 2]              0  
>      Conv2D-12        [[1, 512, 2, 2]]        [1, 512, 2, 2]          2,359,808  
>   BatchNorm2D-12  [[1, 512, 2, 2]]        [1, 512, 2, 2]            2,048  
>       ReLU-12           [[1, 512, 2, 2]]        [1, 512, 2, 2]              0  
>      Conv2D-13        [[1, 512, 2, 2]]        [1, 512, 2, 2]          2,359,808  
>   BatchNorm2D-13  [[1, 512, 2, 2]]        [1, 512, 2, 2]            2,048  
>       ReLU-13           [[1, 512, 2, 2]]        [1, 512, 2, 2]              0  
>     MaxPool2D-5      [[1, 512, 2, 2]]        [1, 512, 1, 1]              0  
> AdaptiveAvgPool2D-1  [[1, 512, 1, 1]]   [1, 512, 7, 7]            0  
>      Linear-1             [[1, 25088]]              [1, 4096]         102,764,544  
>       ReLU-14            [[1, 4096]]                [1, 4096]                0  
>      Dropout-1          [[1, 4096]]                [1, 4096]                0  
>      Linear-2             [[1, 4096]]                [1, 4096]         16,781,312  
>       ReLU-15            [[1, 4096]]                [1, 4096]                0  
>      Dropout-2          [[1, 4096]]                [1, 4096]                0  
>      Linear-3             [[1, 4096]]                [1, 1000]          4,097,000  
> -------------------------------------------------------------------------------
> Total params: 138,374,440
> Trainable params: 138,357,544
> Non-trainable params: 16,896
> -------------------------------------------------------------------------------
> Input size (MB): 0.01
> Forward/backward pass size (MB): 6.95
> Params size (MB): 527.86
> Estimated Total Size (MB): 534.82
> -------------------------------------------------------------------------------
> ```



### 五、训练配置

#### 1）优化器

常用的优化方法：Momentum/AdaGrad/Adam

* 选用的优化器：Momentum

* Momentum基本思想：

	<img src="./image classification with VGG model on CIFAR-100 dataset-image/momentum.png" alt="image-20210804162038098" style="zoom: 67%;" />

	考虑历史梯度，引导参数朝着最优值更快收敛。结合物理学上动量思想，在梯度下降问题中引入动量项v和折扣因子γ，公式变为：
	$$
	v \leftarrow γv+α\nabla_{\Theta}J(\Theta)
	$$

	$$
	\Theta \leftarrow \Theta-v
	$$

	γ表示历史梯度的影响力，γ越大，历史梯度对现在的影响也越大，其中$J(\Theta)$为损失函数。

#### 2）损失函数

交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度，在机器学习中就表示为真实概率分布与预测概率分布之间的差异。交叉熵的值越小，模型预测效果就越好。**交叉熵损失函数**的标准形式如下**:**
$$
L=-\frac{1}{N} \left[ y\ln a+(1-y)\ln (1-a) \right]
$$
公式中$x$表示样本，$y$表示实际的标签， $a$表示预测的输出，$N$表示样本总数量。

多分类问题中的loss函数：
$$
L=-\frac{1}{N} \sum_{i}y_{i}\ln a_{i}
$$

#### 3）代码实现

学习率选取为0.01，折扣因子（动量项）选取为0.9。

```python
def prepare(model):
    #定义优化器、损失函数、精确度计算
    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.01, 	#学习率
        momentum=0.9,		#动量项
        parameters=model.parameters()
    )
    lossFunc = paddle.nn.CrossEntropyLoss() #交叉熵损失函数
    acc = paddle.metric.Accuracy() #计算准确率

    # 设置优化器，损失函数等
    model.prepare(optimizer, lossFunc, acc)
    return model
```



### 六、模型训练与评估

代码实现：

```python
#模型训练
def train(model, train_dataset, epochs=30, batch_size=256):
    callback = paddle.callbacks.VisualDL(log_dir='./log')
    #开始训练
    model.fit(
        train_data=train_dataset, #训练集
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback,
        verbose=1,   #输出详细信息
        shuffle=True #每个epoch打乱顺序
    )
    return model

#模型评估
def evaluate(model, eval_dataset, batch_size=256):
    result = model.evaluate(
        eval_dataset, #测试集
        batch_size=batch_size,
        verbose=1
    )
    return result

#模型保存
def save(model, path):
    model.save(path)

#main()
if __name__ == "__main__":
    train_dataset, eval_dataset = data_process() #获取数据
    visualize(train_dataset) #数据可视化
    vgg16 = VGG(16)		#构建模型
    model = prepare(vgg16) #训练预配置
    model = train(model, train_dataset, epochs=30, batch_size=1024) #训练：30epochs，每batch512个样本
    result = evaluate(model, eval_dataset, batch_size=512)
    print(result)
    save(model,.'/models/vgg') #保存模型
```

- 运行环境：WIN10 64位
- CPU：Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz   1.80 GHz
- RAM 8.00 GB
- 运行方式：单机单卡

训练过程中损失和正确率的变化：

> ```
> The loss value printed in the log is the current step, and the metric is the average value of previous steps.
> Epoch 1/30
> step 49/49 [============] - loss: 4.3501 - acc: 0.0251 - 57s/step  
> Epoch 2/30
> step 49/49 [============] - loss: 4.0491 - acc: 0.0650 - 57s/step  
> Epoch 3/30
> step 49/49 [============] - loss: 3.8438 - acc: 0.1000 - 57s/step  
> Epoch 4/30
> step 49/49 [============] - loss: 3.6927 - acc: 0.1277 - 57s/step  
> Epoch 5/30
> step 49/49 [============] - loss: 3.3667 - acc: 0.1523 - 57s/step  
> Epoch 6/30
> step 49/49 [============] - loss: 3.2304 - acc: 0.1787 - 57s/step  
> Epoch 7/30
> step 49/49 [============] - loss: 3.2907 - acc: 0.1992 - 56s/step  
> Epoch 8/30
> step 49/49 [============] - loss: 3.0326 - acc: 0.2206 - 56s/step  
> Epoch 9/30
> step 49/49 [============] - loss: 2.9636 - acc: 0.2410 - 57s/step  
> Epoch 10/30
> step 49/49 [============] - loss: 2.8917 - acc: 0.2601 - 57s/step  
> Epoch 11/30
> step 49/49 [============] - loss: 2.8274 - acc: 0.2793 - 56s/step  
> Epoch 12/30
> step 49/49 [============] - loss: 2.6850 - acc: 0.2965 - 56s/step  
> Epoch 13/30
> step 49/49 [============] - loss: 2.6051 - acc: 0.3124 - 57s/step  
> Epoch 14/30
> step 49/49 [============] - loss: 2.4702 - acc: 0.3348 - 59s/step  
> Epoch 15/30
> step 49/49 [============] - loss: 2.5172 - acc: 0.3441 - 60s/step  
> Epoch 16/30
> step 49/49 [============] - loss: 2.3454 - acc: 0.3574 - 66s/step  
> Epoch 17/30
> step 49/49 [============] - loss: 2.4128 - acc: 0.3750 - 68s/step  
> Epoch 18/30
> step 49/49 [============] - loss: 2.2570 - acc: 0.3873 - 72s/step  
> Epoch 19/30
> step 49/49 [============] - loss: 2.2628 - acc: 0.4045 - 77s/step  
> Epoch 20/30
> step 49/49 [============] - loss: 2.1107 - acc: 0.4106 - 76s/step  
> Epoch 21/30
> step 49/49 [============] - loss: 2.1189 - acc: 0.4290 - 79s/step  
> Epoch 22/30
> step 49/49 [============] - loss: 2.0846 - acc: 0.4384 - 72s/step  
> Epoch 23/30
> step 49/49 [============] - loss: 2.0241 - acc: 0.4514 - 88s/step  
> Epoch 24/30
> step 49/49 [============] - loss: 1.9438 - acc: 0.4622 - 88s/step  
> Epoch 25/30
> step 49/49 [============] - loss: 1.9774 - acc: 0.4787 - 71s/step  
> Epoch 26/30
> step 49/49 [============] - loss: 1.9579 - acc: 0.4823 - 80s/step  
> Epoch 27/30
> step 49/49 [============] - loss: 1.8866 - acc: 0.4934 - 67s/step  
> Epoch 28/30
> step 49/49 [============] - loss: 1.8058 - acc: 0.5096 - 57s/step  
> Epoch 29/30
> step 49/49 [============] - loss: 1.8005 - acc: 0.5169 - 57s/step  
> Epoch 30/30
> step 49/49 [============] - loss: 1.6318 - acc: 0.5336 - 57s/step  
> Eval begin...
> step 20/20 [============] - loss: 2.3094 - acc: 0.3921 - 8s/step  
> Eval samples: 10000
> {'loss': [2.309418], 'acc': 0.3921}
> ```

可视化损失值及正确率：

<img src="./image classification with VGG model on CIFAR-100 dataset-image/result.png" alt="image-20210806085954577" style="zoom:80%;" />

从损失值和正确率的变化趋势来看，loss始终在下降且为完全收敛，正确率一直在上升，也仍未收敛。从最终的训练集和验证集的正确率来看，模型的正确率较低。由于数据量以及参数量均较大（共一亿参数，500+MB），且本地无CUDA且paddlepaddle高层API在AIstudio运行会崩溃（原因不明），在本地CPU上运行需要很长时间，训练30epoch大约使用了20小时。从损失函数和正确值的变化可推测，损失函数还未完全收敛，正确率应该还能够继续提升，在实际应用中一般需要训练100+epoch，预计正确率还会上升。



### 七、总结

深度学习的万能步骤是数据处理、模型设计、训练配置、开启训练、模型保存、可视化等，结合深度学习框架这些步骤就可通过简单的封装调用来实现。这次实训的主要收获是关于数据增强的处理、batch_norm的归一化、以及通过选取比较适合的优化器、优化器的参数以及合适的batch_size等参数可提高模型训练速度及准确率。实验的不足是由于时间较紧，所以没有使用基础的API而是使用2.0的高层API，大大减少了编码的复杂度，还有因为由于VGG网络参数规模比较庞大，单次训练需要数天，所以只能训练几十个epoch，且难以通过调参对比由于学习率和batch_size大小对模型准确率的影响。

