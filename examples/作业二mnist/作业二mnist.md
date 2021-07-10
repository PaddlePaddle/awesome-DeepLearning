一（二）、损失函数及其代码实现
（1）0-1损失函数
0-1损失是指预测值和目标值不相等为1， 否则为0:
```
def 0_1_loss(y, y_pred):
 if(y==y_pred):
  return 0
 else:
  return 1
```
（2）绝对值损失函数
绝对值损失函数是计算预测值与目标值的差的绝对值
```
def abs_loss(y, y_pred):
 return abs(y_pred - y)
```
（3） log对数损失函数
log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。健壮性不强，相比于hinge loss对噪声更敏感。逻辑回归的损失函数就是log对数损失函数。
```
def log_loss(y, y_pred):
 if true_label == 1:
  return -np.log(y_pred)
 else:
  return -np.log(1 - y_pred)
```
（4）平方损失函数
```
def square_loss(y, y_pred):
 loss = (np.square(y - y_pred)).sum()
 return loss
```
（5）指数损失函数
```
def exp_loss(y , y_pred):
 return math.exp(-y * y_pred)
```
（6）交叉熵损失函数
```
def cross_entropy_error(y,t):
 delta=1e-7  
 return -np.sum(t * np.log(y + delta)）
```
（7）合页损失函数
```
def hinge_loss(y, y_pred):
 return max(0, (1 - y * y_pred))
```

三、池化方法
1. 一般池化（General Pooling）
	
    池化作用于图像中不重合的区域，又包含平均池化（mean pooling，计算图像区域的平均值作为该区域池化后的值）和最大池化（max pooling，选图像区域最大值作为池化后的值)
2. 重叠池化（OverlappingPooling）

  相邻池化窗口之间会有重叠区域
3. 空金字塔池化（Spatial Pyramid Pooling）
  
  空间金字塔池化可以把任何尺度的图像的卷积特征转化成相同维度，这不仅可以让CNN处理任意尺度的图像，还能避免cropping和warping操作，导致一些信息的丢失，具有非常重要的意义。
4. 全局平均池化

  若有四张特征，将每一张特征图计算所有像素点的均值，输出一个数据值，这样4 个特征图就会输出4个数据点，将这些数据点组成一个1 * 4的向量的话，就成为一个特征向量，就可以送入到softmax的分类中计算了。
5. 全局最大池化

  将每张特诊图选取最大值像素点组成向量

四、数据增强方法

有监督增强方法：
1. 几何变换：没有改变图像本身的内容，是选择了图像的一部分或者对像素进行了重分布。翻转，旋转，裁剪，变形，缩放
2. 颜色变换：改便了图像本身内容。噪声、模糊、颜色变换、擦除、填充
3. SMOTE：即Synthetic Minority Over-sampling Technique方法，它是通过人工合成新样本来处理样本不平衡问题，从而提升分类器性能。SMOTE方法是基于插值的方法，它可以为小样本类合成新的样本，主要流程为：

第一步，定义好特征空间，将每个样本对应到特征空间中的某一点，根据样本不平衡比例确定好一个采样倍率N；

第二步，对每一个小样本类样本(x,y)，按欧氏距离找出K个最近邻样本，从中随机选取一个样本点，假设选择的近邻点为(xn,yn)。在特征空间中样本点与最近邻样本点的连线段上随机选取一点作为新样本点，满足以下公式：<img src="https://pic3.zhimg.com/50/v2-e49771f0cf504363a2a25f70320c2102_hd.jpg?source=1940ef5c" data-caption="" data-size="normal" data-rawwidth="1060" data-rawheight="78" class="origin_image zh-lightbox-thumb" width="1060" data-original="https://pic3.zhimg.com/v2-e49771f0cf504363a2a25f70320c2102_r.jpg?source=1940ef5c"/>

第三步，重复以上的步骤，直到大、小样本数量平衡。

4. SamplePairing：从训练集中随机抽取两张图片分别经过基础数据增强操作(如随机翻转等)处理后经像素以取平均值的形式叠加合成一个新的样本，标签为原样本标签中的一种。这两张图片甚至不限制为同一类别，这种方法对于医学图像比较有效。经SamplePairing处理后可使训练集的规模从N扩增到N×N。实验结果表明，因SamplePairing数据增强操作可能引入不同标签的训练样本，导致在各数据集上使用SamplePairing训练的误差明显增加，而在验证集上误差则有较大幅度降低。
5. mixup：mixup是Facebook人工智能研究院和MIT在“Beyond Empirical Risk Minimization”中提出的基于邻域风险最小化原则的数据增强方法，它使用线性插值得到新样本数据。令(xn,yn)是插值生成的新数据，(xi,yi)和(xj,yj)是训练集随机选取的两个数据，则数据生成方式如下<img src="https://pic3.zhimg.com/50/v2-1227a3b22b5d242f9ea3706e46b65dd1_hd.jpg?source=1940ef5c" data-caption="" data-size="normal" data-rawwidth="628" data-rawheight="76" class="origin_image zh-lightbox-thumb" width="628" data-original="https://pic1.zhimg.com/v2-1227a3b22b5d242f9ea3706e46b65dd1_r.jpg?source=1940ef5c"/>

λ的取指范围介于0到1。提出mixup方法的作者们做了丰富的实验，实验结果表明可以改进深度学习模型在ImageNet数据集、CIFAR数据集、语音数据集和表格数据集中的泛化误差，降低模型对已损坏标签的记忆，增强模型对对抗样本的鲁棒性和训练生成对抗网络的稳定性。

无监督增强方法：
1. GAN关于GAN(generative adversarial networks)，我们已经说的太多了。它包含两个网络，一个是生成网络，一个是对抗网络，基本原理如下：

(1) G是一个生成图片的网络，它接收随机的噪声z，通过噪声生成图片，记做G(z) 。

(2) D是一个判别网络，判别一张图片是不是“真实的”，即是真实的图片，还是由G生成的图片。

2. Autoaugmentation[5]AutoAugment是Google提出的自动选择最优数据增强方案的研究，这是无监督数据增强的重要研究方向。它的基本思路是使用增强学习从数据本身寻找最佳图像变换策略，对于不同的任务学习不同的增强方法，流程如下：

(1) 准备16个常用的数据增强操作。

(2) 从16个中选择5个操作，随机产生使用该操作的概率和相应的幅度，将其称为一个sub-policy，一共产生5个sub-polices。

(3) 对训练过程中每一个batch的图片，随机采用5个sub-polices操作中的一种。

(4) 通过模型在验证集上的泛化能力来反馈，使用的优化方法是增强学习方法。

(5) 经过80~100个epoch后网络开始学习到有效的sub-policies。

(6) 之后串接这5个sub-policies，然后再进行最后的训练。

五、 图像分类方法综述
1. 基于色彩特征的索引技术

  色彩是物体表面的一种视觉特性,每种物体都有其特有的色彩特征,譬如人们说到绿色往往是和树木或草原相关,谈到蓝色往往是和大海或蓝天相关,同一类物体往拍几有着相似的色彩特征,因此我们可以根据色彩特征来区分物体.用色彩特特征进行图像分类一可以追溯到Swain和Ballard提出的色彩直方图的方法.由于色彩直方图具有简单且随图像的大小、旋转变化不敏感等特点,得到了研究人员的厂泛关注,目前几乎所有基于内容分类的图像数据库系统都把色彩分类方法作为分类的一个重要手段,并提出了许多改进方法,归纳起主要可以分为两类：全局色彩特征索引和局部色彩特征索引。

2. 基于纹理的图像分类技术

  纹理特征也是图像的重要特征之一,其本质是刻画象素的邻域灰度空间分布规律由于它在模式识别和计算机视觉等领域已经取得了丰富的研究成果,因此可以借用到图像分类中。在70年代早期,Haralick等人提出纹理特征的灰度共生矩阵表示法(eo一oeeurrenee matrix representation),这个方法提取的是纹理的灰度级空间相关性(gray level Spatial dependenee),它首先基于象素之间的距离和方向建立灰度共生矩阵,再由这个矩阵提取有意义的统计量作为纹理特征向量。基于一项人眼对纹理的视觉感知的心理研究,Tamuar等人提出可以模拟纹理视觉模型的6个纹理属性,分别是粒度,对比度,方向性,线型,均匀性和粗糙度。QBIC系统和MARS系统就采用的是这种纹理表示方法。
在90年代初期,当小波变换的理论结构建一认起来之后,许多研究者开始研究
如何用小波变换表示纹理特征。smiht和chang利用从小波子带中提取的统计量(平均值和方差)作为纹理特征。这个算法在112幅Brodatz纹理图像中达到了90%的准确率。为了利用中间带的特征,Chang和Kuo开发出一种树型结构的小波变化来进一步提高分类的准确性。还有一些研究者将小波变换和其他的变换结合起来以得到更好的性能,如Thygaarajna等人结合小波变换和共生矩阵,以兼顾基于统计的和基于变换的纹理分析算法的优点。

3. 基于形状的图像分类技术

  形状是图像的重要可视化内容之一在二维图像空间中,形状通常被认为是一条封闭的轮廓曲线所包围的区域,所以对形状的描述涉及到对轮廓边界的描述以及对这个边界所包围区域的描述.目前的基于形状分类方法大多围绕着从形状的轮廓特征和形状的区域特征建立图像索引。关于对形状轮廓特征的描述主要有:直线段描述、样条拟合曲线、傅立叶描述子以及高斯参数曲线等等。
实际上更常用的办法是采用区域特征和边界特征相结合来进行形状的相似分类.如Eakins等人提出了一组重画规则并对形状轮廓用线段和圆弧进行简化表达,然后定义形状的邻接族和形族两种分族函数对形状进行分类.邻接分族主要采用了形状的边界信息,而形状形族主要采用了形状区域信息.在形状进行匹配时,除了每个族中形状差异外,还比较每个族中质心和周长的差异,以及整个形状的位置特征矢量的差异,查询判别距离是这些差异的加权和。

4. 基于空间关系的图像分类技术

  在图像信息系统中,依据图像中对象及对象间的空间位置关系来区别图像库中的不同图像是一个非常重要的方法。因此,如何存贮图像对象及其中对象位置关系以方便图像的分类,是图像数据库系统设计的一个重要问题。而且利用图像中对象间的空间关系来区别图像,符合人们识别图像的习惯,所以许多研究人员从图像中对象空间位置关系出发,着手对基于对象空间位置关系的分类方法进行了研究。早在1976年,Tanimoto提出了用像元方法来表示图像中的实体,并提出了用像元来作为图像对象索引。随后被美国匹兹堡大学chang采纳并提出用二维符号串(2D一String)的表示方法来进行图像空间关系的分类,由于该方法简单,并且对于部分图像来说可以从ZD一String重构它们的符号图,因此被许多人采用和改进,该方法的缺点是仅用对象的质心表示空间位置;其次是对于一些图像来说我们不能根据其ZD一string完个重构其符号图;再则是上述的空间关系太简单,实际中的空间关系要复杂得多。,针对这些问题许多人提出了改进力一法。Jungert根据图像对象的最小包围盒分别在:x轴方向和y轴上的投影区间之间的交叠关系来表示对象之间的空间关系,随后Cllallg和Jungert等人又提出了广义ZD一string(ZDG一String)的方法,将图像对象进一步切分为更小的子对象来表示对象的空间关系,但是该方法不足之处是当图像对象数日比较多且空间关系比较复杂时,需要切分的子对象的数目很多,存储的开销太大,针对此Lee和Hsu等人提出了ZDC一string的方一法,它们采用Anell提出的13种时态间隔关系并应用到空间投影区问上来表达空间关系。在x轴方向和y轴方向的组合关系共有169种,他提出了5种基本关系转换法则,在此基础上又提出了新的对象切分方法。采用

  ZDC一string的方法比ZDG一string切分子对象的数目明显减少。为了在空间关系中保留两个对象的相对空间距离和对象的大小,Huang等人提出了ZDC书string的方法提高符号图的重构精度,并使得对包含对象相对大小、距离的符号图的推理成为可能。上述方法都涉及到将图像对象进行划分为子对象,且在用符号串重构对象时处理时间的开销都比较大,为解决这些方法的不足,Lee等人又提出了ZDB一String的方法,它不要求对象进一步划分,用对象的名称来表示对象的起点和终点边界。为了解决符号图的重构问题,Chin一ChenCllang等人提出了面向相对坐标解决符号图的重构问题,Chin一ChenChang等人提出了面向相对坐标符号串表示(RCOS串),它们用对象最小外接包围盒的左下角坐标和右上角坐标来表示对象之间的空间关系.
对于对象之间的空间关系采用Allen提出的13种区间表示方法。实际上上述所有方法都不是和对象的方位无关,为此Huang等人又提出了RSString表示方法。虽然上述各种方法在对图像对象空间信息的分类起到过一定作用,由于它们都是采用对象的最小外接矩形来表示一个对象空间位置,这对于矩形对象来说是比较合适的,但是当两个对象是不规则形状,且它们在空间关系上是分离时,它们的外接矩形却存在着某种包含和交叠,结果出现对这些对象空间关系的错误表示。用上述空间关系进行图像分类都是定性的分类方一法,将图像的空间关系转换为图像相似性的定量度量是一个较为困难的事情。Nabil综合ZD一String方法和二维平面中对象之间的点集拓扑关系。提出了ZD一PIR分类方法,两个对象之间的相似与否就转换为两个图像的ZD一PIR图之间是否同构。ZD一PIR中只有图像对象之间的空间拓扑关系具有旋转不变性,在进行图像分类的时候没有考虑对象之间的相对距离。



在实际应用中，保存到本地的数据存储格式多种多样，如MNIST数据集以json格式存储在本地，其数据存储结构如 图2 所示。

![](https://ai-studio-static-online.cdn.bcebos.com/975f614a1b9e48cd8df65da97c770ec293de8c5eb4c14bcd820e33740a4a8249)


图2：MNIST数据集的存储结构


data包含三个元素的列表：train_set、val_set、 test_set，包括50 000条训练样本、10 000条验证样本、10 000条测试样本。每个样本包含手写数字图片和对应的标签。

train_set（训练集）：用于确定模型参数。
val_set（验证集）：用于调节模型超参数（如多个网络结构、正则化权重的最优选择）。
test_set（测试集）：用于估计应用效果（没有在模型中应用过的数据，更贴近模型在真实场景应用的效果）。
train_set包含两个元素的列表：train_images、train_labels。

train_images：[50 000, 784]的二维列表，包含50 000张图片。每张图片用一个长度为784的向量表示，内容是28*28尺寸的像素灰度值（黑白图片）。
train_labels：[50 000, ]的列表，表示这些图片对应的分类标签，即0~9之间的一个数字。


```python
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np
import paddle.fluid as fluid

# 定义数据集读取器
def load_data(mode='train'):

    # 读取数据文件
    datafile = './data/data17155/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

```

  卷积神经网络的结构多种多样，可以在网络的深度上进行延仲，也可在网络的宽度上进行拓展..CoogleNel采川了多个Inception模块来提升网络的深度和+度，从而达到提高分类准确率，本实验所用的网络是GoogLeNet的简化版。

网络中的 Inception模块由4个分支组成，其具体结构如图所示，输入数据分别由4个分支进行处理（处理前后图像尺寸一样)，然后将4个分支的输出堆叠在一起作为下一层的输入。
![](https://ai-studio-static-online.cdn.bcebos.com/d47d872c562c47b1963c12157f6474adadf2538e04d24da384229f8491fd1fd4)

![](https://ai-studio-static-online.cdn.bcebos.com/80e144d1f620428183fbf331283f86016c070f49f1eb4ca58f4142fe648b1421)



```python
class InceptionA(paddle.nn.Layer):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2D(in_channels, 16, kernel_size=1)  

        self.branch5x5_1 = Conv2D(in_channels, 16, kernel_size=1)  
        self.branch5x5_2 = Conv2D(16, 24, kernel_size=5, padding=2)  

        self.branch3x3_1 = Conv2D(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = Conv2D(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = Conv2D(24, 24, kernel_size=3, padding=1)

        self.branch_pool = Conv2D(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]  
        cat = fluid.layers.concat(outputs, axis=1)
        cat = fluid.layers.relu(cat)
        return cat
```


```python
class Net(paddle.nn.Layer):  
    def __init__(self, name_scope):
        super(Net, self).__init__(name_scope)
        name_scope = self.full_name()
        self.conv1 = Conv2D(1, 10, kernel_size=5, stride=1, padding=0)
        self.conv2 = Conv2D(88, 20, kernel_size=5,stride=1, padding=0)
        
        self.incep1 = InceptionA(in_channels=10)  
        self.incep2 = InceptionA(in_channels=20) 

        self.maxpool = MaxPool2D(kernel_size=2)

        self.fc = Linear(1408, 10)  

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.incep1(x) 
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.incep2(x)  
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        x = F.softmax(x)
        return x
        
```


```python
model = Net("mnist")

with fluid.dygraph.guard():
    
    
    #调用加载数据的函数
    train_loader = load_data('train')
    #选择优化算法
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        correct = 0
        total = 0
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            
            
            total += label.shape[0]
            pred = predict.argmax(1)
            for i in range(len(pred)):
                if(pred[i] == label[i]):
                    correct += 1
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
        print(correct/total)

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist')
```

    loading mnist dataset from ./data/data17155/mnist.json.gz ......
    epoch: 0, batch: 0, loss is: [4.1233544]
    epoch: 0, batch: 200, loss is: [0.15472116]
    epoch: 0, batch: 400, loss is: [0.21236697]
    0.8983
    epoch: 1, batch: 0, loss is: [0.07024439]
    epoch: 1, batch: 200, loss is: [0.0659261]
    epoch: 1, batch: 400, loss is: [0.12282111]
    0.96662
    epoch: 2, batch: 0, loss is: [0.04835995]
    epoch: 2, batch: 200, loss is: [0.07213611]
    epoch: 2, batch: 400, loss is: [0.00998662]
    0.9758
    epoch: 3, batch: 0, loss is: [0.12734409]
    epoch: 3, batch: 200, loss is: [0.03001437]
    epoch: 3, batch: 400, loss is: [0.00621022]
    0.98016
    epoch: 4, batch: 0, loss is: [0.0901669]
    epoch: 4, batch: 200, loss is: [0.03810026]
    epoch: 4, batch: 400, loss is: [0.01157603]
    0.98298



```python
valid_loader = load_data('valid')

correct = 0
total = 0
for batch_id, data in enumerate(valid_loader()):
     #准备数据，变得更加简洁
    image_data, label_data = data
    image = fluid.dygraph.to_variable(image_data)
    label = fluid.dygraph.to_variable(label_data)
               
    predict = model(image)
                        
    total += label.shape[0]
    pred = predict.argmax(1)
    for i in range(len(pred)):
        if(pred[i] == label[i]):
            correct += 1   
print(correct/total)
```

    loading mnist dataset from ./data/data17155/mnist.json.gz ......
    0.9829


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
