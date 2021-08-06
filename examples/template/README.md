# 1、背景知识

## 1.1 、VGG概念

vgg网络模型于2014年由牛津大学著名研究组Oxford Visual Geometry Group提出，该网络的亮点在于采用重复堆叠的3*3的小卷积核替代大卷积核、2x2的小池化核，在保证具有相同感受野的条件下，层数更深特征图更宽，提升了网络的深度，从而提升网络特征提取的能力。

## 1.2、 VGG原理

采用连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）。对于给定的感受野（与输出有关的输入图片的局部大小），采用堆积的小卷积核是优于采用大的卷积核，因为多层非线性层可以增加网络深度来保证学习更复杂的模式，而且代价还比较小（参数更少）。

本次实训使用的网络模型是vgg11，网络结构如图：

![img](../../../../计科1803-201878020116-潘浩翔暑假实习与实验报告/images/4b58c73a442d47eea19dab77baa342a1c8c3ca643b6f41c496b8ff3d9384f3a4)

VGG-11有8层卷积和3层全连接层。VGG网络的设计严格使用3×3的卷积层和池化层来提取特征，并在网络的最后面使用三层全连接层，将最后一层全连接层的输出作为分类的预测。 在VGG中每层卷积将使用ReLU作为激活函数，在全连接层之后添加dropout来抑制过拟合。使用小的卷积核能够有效地减少参数的个数，使得训练和测试变得更加有效。比如使用两层3×3卷积层，可以得到感受野为5的特征图，而比使用5×5的卷积层需要更少的参数。由于卷积核比较小，可以堆叠更多的卷积层，加深网络的深度，这对于图像分类任务来说是有利的。VGG模型的成功证明了增加网络的深度，可以更好的学习图像中的特征模式。

## 1.3 、参考论文

[Simonyan K, Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition[J\]. Computer Science, 2014.](https://aistudio.baidu.com/aistudio/projectdetail/"https://arxiv.org/pdf/1409.1556.pdf")

# 2 代码实现

## 2.1 实验设计逻辑

实验任务为使用Cifar100数据集，基于vgg网络实现图像分类

## 2.2 数据准备

### 2.2.1 Cifar100数据集介绍

Cifar100数据集由100个类的60000个32x32彩色图像组成，每个类有600个图像，每类有500个训练图像和100个测试图像，这100个类又被分为20个超类。

使用到的数据文件是train、test和meta。

train和test文件结构一样，均由数据和标签组成，数据内容是50000x3072 uint8的numpy数组（test数据集为10000x3072 ），阵列的每一行存储32x32彩色图像，即每一行存储32*32*3=3072个数字信息，每1024个数据条目包含一个通道值，存储顺序为红绿蓝；标签为范围为0-9的数字列表。

meta文件中包含fine_label_names和coarse_label_names两个列表，fine_label_names中每个标签的索引与train和test文件中的标签一一对应

### 2.2.2 加载训练集和测试集

飞桨框架已收录Cifar100数据集，可使用API快速加载数据集。 使用方法为paddle.vision.datasets.Cifar100，设置参数的功能介绍如下： (1)data_file：本地数据集路径，默认为None，自动下载 (2)mode：’train’加载训练集，’test’加载测试集 (3)transform：图片数据预处理 (4)download：为True时自动下载数据集 设置数据集路径，传入设定的transform，设置mode参数为test或train就可以按一致的数据处理方法加载训练集和测试集

```
    transform = T.Compose([
                T.Resize((96,96)),
                T.Transpose((2,0,1)),
                # 数据的格式转换和标准化 CHW => HCW 图像归一化 
                T.Normalize(mean=[127.5], std=[127.5], data_format='HWC', to_rgb=True)  # 
            ])
    # 通过transform参数传递定义好的数据增项方法即可完成对自带数据集的应用
    # 训练数据集
    train_dataset = paddle.vision.datasets.Cifar100(data_file='/home/aistudio/realize/cifar-100-python.tar.gz', 
        download=False, mode='train', transform=transform)
    # 验证数据集
    eval_dataset = paddle.vision.datasets.Cifar100(data_file='/home/aistudio/realize/cifar-100-python.tar.gz', 
        download=False, mode='test', transform=transform)
```

### 2.2.3 加载分类标签

读取meta文件，设置编码格式为latin1加载文件中的超类和类信息

```
file_obj = open('/home/aistudio/cifar-100-python/meta','rb')
meta_labelNames = pickle.load(file_obj, encoding='latin1')
```

## 2.3 模型设计

使用飞桨PaddlePaddle2.0高层API搭建vgg11网络模型

```
    vgg11_cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    vgg11_features = make_layers(vgg11_cfg)
    # build vgg11 vgg11_model，输出为100个分类的权重
    vgg11_net = paddle.vision.models.VGG(features=vgg11_features,num_classes=100)

    inputType = paddle.static.InputSpec([None, 3, 96, 96], 'float32', 'image')
    labelType = paddle.static.InputSpec([None, 1], 'int64', 'label')

    # 使用高层API——paddle.Model将网络结构用 Model类封装成为模型
    vgg11_model = paddle.Model(vgg11_net,inputType,labelType)
```

## 2.4 训练配置

配置模型所需的部件，为模型训练做准备，设置Adam优化器，可利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率，学习率learning_rate设置为5e-4，设置L2正则化方法减少过拟合现象的产生；设置计算输入input和标签label间的交叉熵损失函数为paddle.nn.CrossEntropyLoss，该函数结合 LogSoftmax 和 NLLLoss 的OP计算用于训练n 类分类器；评价指标选择top1和top5的验证正确率。

```
#为模型训练做准备，设置优化器，损失函数和评价指标
vgg11_model.prepare(
    optimizer=paddle.optimizer.Adam(
		learning_rate=5e-4, weight_decay=paddle.regularizer.L2Decay(5e-4), 	parameters=vgg11_model.parameters()),
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy(topk=(1, 5))
)
```

## 2.5 应用部署

设置模型保存目录后进行训练，选择训练效果最好的版本，Model.load加载训练好的模型。

```
#保存训练过程中的模型作为备选
vgg11_model.fit(
...
 save_dir='/home/aistudio/work/vgg11',
...
}
#训练结束再保存一次
vgg11_model.save('/home/aistudio/work/vgg11')
#加载训练模型
vgg11_model.load('/home/aistudio/work/vgg11/final)
```

## 2.6 模型训练与评估

paddle.Model对模型进行了封装，调用高层API接口Model.fit()来启动训练过程，调用Model.evaluate()指定数据集即可启动模型评估。

```
# 加载数据集
transform = T.Compose([
            T.Resize((96,96)),
            T.Transpose((2,0,1)),
            # 数据的格式转换和标准化 CHW => HCW 图像归一化 
            T.Normalize(mean=[127.5], std=[127.5], data_format='HWC', to_rgb=True)  # 
        ])
# 通过transform参数传递定义好的数据增项方法即可完成对自带数据集的应用
# 训练数据集
train_dataset = paddle.vision.datasets.Cifar100(data_file='/home/aistudio/realize/cifar-100-python.tar.gz', 
    download=False, mode='train', transform=transform)
# 验证数据集
eval_dataset = paddle.vision.datasets.Cifar100(data_file='/home/aistudio/realize/cifar-100-python.tar.gz', 
    download=False, mode='test', transform=transform)

# 模型训练
def train(vgg11_model=None):
    global transform,train_dataset,eval_dataset
    vgg11_model.fit(
        train_dataset,
        eval_dataset,
        epochs=100,
        batch_size=128,
        log_freq=100,
        verbose=1
    )
    return vgg11_model
# 模型评估
def evaluate(vgg11_model=None):
    global transform,train_dataset,eval_dataset
    print('----------使用训练集评估...')
    train_dataset_eval_result=vgg11_model.evaluate(train_dataset, verbose=0)
    print('----------训练集评估完成，使用测试集评估...')
    test_dataset_eval_result=vgg11_model.evaluate(eval_dataset, verbose=0)
    print('-----------------------训练集评估结果-----------------------')
    print(train_dataset_eval_result)
    print('-----------------------测试集评估结果-----------------------')
    print(test_dataset_eval_result)
```

## 2.7 模型推理

### 2.7.1 自定义数据集合

paddle可以通过继承paddle.io.Dataset将任意一个待测样本转换为Model可接受的数据集

```
#继承paddle.io.Dataset类
class MyDataset(paddle.io.Dataset):
#实现构造函数，定义数据集处理读取方式
    def __init__(self, image, label, transform=None):
        super(MyDataset, self).__init__()
        self.label = label
        self.img = image
        self.transform = transform
    def __getitem__(self, index):  # 返回图片数据和标签
        return self.transform(self.img), np.array(self.label).astype('int64')
    def __len__(self):  # 返回数据集的长度为1
        return 1
#返回数据集
def getDataset(path=None,label=-1):
    transform = T.Compose([
            T.Resize((96,96)),
            T.Transpose((2,0,1)),
            # 数据的格式转换和标准化 CHW => HCW 图像归一化 
            T.Normalize(mean=[127.5], std=[127.5], data_format='HWC', to_rgb=True) 
        ])
    img = Image.open(path)
    return MyDataset(image=np.array(img), label=label ,transform=transform)
```

### 2.7.2 模型推理

paddle.Model对模型进行了封装，调用高层API接口Model.predict()进行推理。

```
#输出推理结果
def print_result(num=None,label=-1):
    file_obj = open('/home/aistudio/realize/meta','rb')
    data = pickle.load(file_obj,encoding='latin1')
    coarse_label_names=['aquatic_mammals', 'fish', 'flowers', 'food_containers', 
    'fruit_and_vegetables', 'household_electrical_devices', 'household_furniture', 
    'insects', 'large_carnivores', 'large_man-made_outdoor_things', 
    'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores', 
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 
    'small_mammals', 'trees', 'vehicles_1', 'vehicles_2']
    core_final={
        'aquatic_mammals' : ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
        'fish' : ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
        'flowers' : ['orchid', 'poppy',  'rose', 'sunflower', 'tulip'],
        'food_containers' : ['bottle', 'bowl', 'can', 'cup', 'plate'],
        'fruit_and_vegetables' : ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
        'household_electrical_devices' : ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
        'household_furniture' : ['bed', 'chair', 'couch', 'table', 'wardrobe'],
        'insects' : ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
        'large_carnivores' : ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
        'large_man-made_outdoor_things' : ['bridge', 'castle', 'house', 'road', 'skyscraper'],
        'large_natural_outdoor_scenes' : ['cloud', 'forest', 'mountain', 'plain', 'sea'],
        'large_omnivores_and_herbivores' : ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
        'medium_mammals' : ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
        'non-insect_invertebrates' : ['crab', 'lobster', 'snail', 'spider', 'worm'],
        'people' : ['baby', 'boy', 'girl', 'man', 'woman'],
        'reptiles' : ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
        'small_mammals' : ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
        'trees' : ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
        'vehicles_1' : ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
        'vehicles_2' : ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
    }
    numtmp = num.copy()
    finalInedx=[]
    coreIndex=[]
    Inf = min(numtmp)-1
    for i in range(5):
        tmp=numtmp.index(max(numtmp))
        finalInedx.append(tmp)
        for j in range(20):
            list1 = core_final[coarse_label_names[j]]
            index = list1.index(data['fine_label_names'][tmp]) if (data['fine_label_names'][tmp] in list1) else -1
            if(index!=-1):
                coreIndex.append(j)
                break
        numtmp[numtmp.index(max(numtmp))]=Inf
    
    realCore=''
    for i in range(20):
        list1 = core_final[coarse_label_names[i]]
        index = list1.index(data['fine_label_names'][label]) if (data['fine_label_names'][label] in list1) else -1
        if(index!=-1):
            realCore=coarse_label_names[i]

    print('----------原始图片标签： %d 分类：%s 超类: %s'%(label,data['fine_label_names'][label],realCore))
    print("----------top1 权重：%lf 标签：%d 分类：%s 超类：%s"
        %(num[finalInedx[0]],finalInedx[0],data['fine_label_names'][finalInedx[0]],coarse_label_names[coreIndex[0]]))
    print("----------top5")
    for i in range(5):
        print("----------权重：%lf 标签：%d 分类：%s 超类：%s"
            %(num[finalInedx[i]],finalInedx[i],data['fine_label_names'][finalInedx[i]],coarse_label_names[coreIndex[i]]))
# 模型推理
def predict(vgg11_model=None,sample=None):
    realLabel=sample[0][1]
    sample=paddle.io.DataLoader(sample, places=paddle.CPUPlace())
    print(vgg11_model.evaluate(sample, verbose=0))
    result = vgg11_model.predict(sample)
    print_result(result[0][0][0].tolist(),realLabel)
```

# 3、 实验总结

以前学过CNN，从以前的基础上手VGG相对来说容易一说。通过这个过程了解了其基本原理，学习到了平常没有学习到的知识，对能力的提升有显著的提高。
