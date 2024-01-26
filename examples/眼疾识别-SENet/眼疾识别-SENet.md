# SENet模型分类眼疾识别数据集
### 数据集准备

/home/aistudio/data/data19065 目录包括如下三个文件，解压缩后存放在/home/aistudio/data/palm目录下。
- training.zip：包含训练中的图片和标签
- validation.zip：包含验证集的图片
- valid_gt.zip：包含验证集的标签


```
!unzip -o -q -d /home/aistudio/data/palm /home/aistudio/data/data19469/training.zip
%cd /home/aistudio/data/palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q -d /home/aistudio/data/palm /home/aistudio/data/data19469/validation.zip
!unzip -o -q -d /home/aistudio/data/palm /home/aistudio/data/data19469/valid_gt.zip
```

    /home/aistudio/data/palm/PALM-Training400


### 定义数据读取器

使用OpenCV从磁盘读入图片，将每张图缩放到$224\times224$大小，并且将像素值调整到$[-1, 1]$之间，代码如下所示：


```
import cv2
import os
import random
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
from paddle.fluid.dygraph.base import to_variable

DATADIR = '/home/aistudio/data/palm/PALM-Training400/PALM-Training400'
DATADIR2 = '/home/aistudio/data/palm/PALM-Validation400'
CSVFILE = '/home/aistudio/labels.csv'

def transform_img(img):
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    img = img / 255.
    img = img * 2.0 - 1.0
    return img


def data_loader(datadir, batch_size=10, mode = 'train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            if name[0] == 'H' or name[0] == 'N':
                label = 0
            elif name[0] == 'P':
                label = 1
            else:
                raise('Not excepted file name')
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader

def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    filelists = open(csvfile).readlines()
    def reader():
        batch_imgs = []
        batch_labels = []
        for line in filelists[1:]:
            line = line.strip().split(',')
            name = line[1]
            label = int(line[2])
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield imgs_array, labels_array

    return reader
```

### SENet网络模型

在深度学习领域，CNN分类网络的发展对其它计算机视觉任务如目标检测和语义分割都起到至关重要的作用，因为检测和分割模型通常是构建在CNN分类网络（称为backbone）之上。提到CNN分类网络，我们所熟知的是VGG，ResNet，Inception，DenseNet等模型，它们的效果已经被充分验证，而且被广泛应用在各类计算机视觉任务上。这里我们使用的是SENet模型，它赢得了最后一届ImageNet 2017竞赛分类任务的冠军。重要的一点是SENet思路很简单，很容易扩展在已有网络结构中。

SENet的基本结构如图
![](https://ai-studio-static-online.cdn.bcebos.com/86433f577c784c15b98560a7d5dad101b300b923331b497a9458fdedfa7e48f3)

原来的任意变换，将输入X变为输出U，现在，假设输出的U不是最优的，每个通道的重要程度不同，有的通道更有用，有的通道则不太有用。

对于每一输出通道，先global average pool，每个通道得到1个标量，C个通道得到C个数，然后经过FC-ReLU-FC-Sigmoid得到C个0到1之间的标量，作为通道的权重，然后原来的输出通道每个通道用对应的权重进行加权（对应通道的每个元素与权重分别相乘），得到新的加权后的特征，作者称之为feature recalibration。

第一步每个通道HxW个数全局平均池化得到一个标量，称之为Squeeze，然后两个FC得到01之间的一个权重值，对原始的每个HxW的每个元素乘以对应通道的权重，得到新的feature map，称之为Excitation。任意的原始网络结构，都可以通过这个Squeeze-Excitation的方式进行feature recalibration，采用了改方式的网络，即SENet版本。

在Inception和ResNet模块上使用了se模块

![](https://ai-studio-static-online.cdn.bcebos.com/4b6a886cb54e4a75b19c674088401e94409eaa278d0b4feca7b25496d3842ef7)

上述的se模块的顺序是global average pooling-FC-ReLU-FC-Sigmoid过程。r是中间隐藏状态特征的维度，在后面也就是了调参实验。最后用了Sigmoid函数确保了权重大小在(0,1)之间。




```
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear, AvgPool2D
import paddle.nn.functional as F
import paddle.nn as nn
import math

class BasicBlock(nn.Layer):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2D(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2D(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2D(planes//16, planes, kernel_size=1)
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        aa = out.shape[2]
        w = F.avg_pool2d(out, aa)
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Layer):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2D(in_planes)
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, planes, kernel_size=1, stride=stride)
            )

        # SE layers
        self.fc1 = nn.Conv2D(planes, planes//16, kernel_size=1)
        self.fc2 = nn.Conv2D(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        
        aa = out.shape[2]
        w = F.avg_pool2d(out, aa)
        w = self.fc1(w)
        w = F.relu(w)
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(25088, num_classes)
        stdv = 1.0/math.sqrt(25088 * 1.0)
        self.out = paddle.fluid.dygraph.nn.Linear(input_dim=10, output_dim=1,param_attr=fluid.param_attr.ParamAttr(initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        out = self.out(out)
        return out


def SENet18():
    return SENet(PreActBlock, [2,2,2,2])
```

### 训练模型并在验证集上进行验证


```
with fluid.dygraph.guard():
    model = SENet18()
    print('start training ... ')
    model.train()
    epoch_num = 3
    # 定义优化器
    opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameter_list=model.parameters())
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batch_size=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            logits = model(img)
            # 进行loss计算
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)  #loss的目的是让sigmoid(logits)去逼近label 所以在预测的时候预测值是sigmoid(logits) 
            avg_loss = fluid.layers.mean(loss)
            if batch_id % 10 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()
            
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = fluid.dygraph.to_variable(x_data)
            label = fluid.dygraph.to_variable(y_data)
            # 运行模型前向计算，得到预测值
            logits = model(img)
            # 二分类，sigmoid计算后的结果以0.5为阈值分两个类别
            # 计算sigmoid后的预测概率，进行loss计算
            pred = fluid.layers.sigmoid(logits)## 这个值大余）0.5就代表预测值为1
            loss = fluid.layers.sigmoid_cross_entropy_with_logits(logits, label)
            pred2 = pred * (-1.0) + 1.0
            # 得到两个类别的预测概率，并沿第一个维度级联
            pred = fluid.layers.concat([pred2, pred], axis=1) # [10，2]
            acc = fluid.layers.accuracy(pred, fluid.layers.cast(label, dtype='int64'))
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
        model.train()
    # save params of model
    fluid.save_dygraph(model.state_dict(), 'palm')
    # save optimizer state
    fluid.save_dygraph(opt.state_dict(), 'palm')
```

    start training ... 
    epoch: 0, batch_id: 0, loss is: [0.6937493]
    epoch: 0, batch_id: 10, loss is: [0.6856425]
    epoch: 0, batch_id: 20, loss is: [0.63147783]
    epoch: 0, batch_id: 30, loss is: [0.5303859]
    [validation] accuracy/loss: 0.9200000762939453/0.31031665205955505
    epoch: 1, batch_id: 0, loss is: [0.27883127]
    epoch: 1, batch_id: 10, loss is: [0.3297333]
    epoch: 1, batch_id: 20, loss is: [0.05639496]
    epoch: 1, batch_id: 30, loss is: [0.3844355]
    [validation] accuracy/loss: 0.9475000500679016/0.17045293748378754
    epoch: 2, batch_id: 0, loss is: [0.09746314]
    epoch: 2, batch_id: 10, loss is: [0.23634449]
    epoch: 2, batch_id: 20, loss is: [0.61998594]
    epoch: 2, batch_id: 30, loss is: [0.04771285]
    [validation] accuracy/loss: 0.9524999856948853/0.11882917582988739

