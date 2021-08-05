# 解压数据集

In [ ]

```
# 解压，若已解压则忽略
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19469/training.zip
%cd /home/aistudio/work/palm/PALM-Training400/
!unzip -o -q PALM-Training400.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19469/validation.zip
!unzip -o -q -d /home/aistudio/work/palm /home/aistudio/data/data19469//valid_gt.zip
/home/aistudio/work/palm/PALM-Training400
```

In [ ]

```
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries

```

# DPN算法介绍

DPN和ResNeXt（ResNet）的结构很相似。最开始一个7*7的卷积层和max pooling层，然后是4个stage，每个stage包含几个sub-stage（后面会介绍），再接着是一个global average pooling和全连接层，最后是softmax层。重点在于stage里面的内容，也是DPN算法的核心。

![img](https://ai-studio-static-online.cdn.bcebos.com/5cb81ea9b6cf40f29bad74320fb5d47e7793ff58cc514c79bc903579b6cbf482)

因为DPN算法简单讲就是将ResNeXt和DenseNet融合成一个网络，因此在介绍DPN的每个stage里面的结构之前，先简单过一下ResNet（ResNeXt和ResNet的子结构在宏观上是一样的）和DenseNet的核心内容。下图中的（a）是ResNet的某个stage中的一部分。（a）的左边竖着的大矩形框表示输入输出内容，对一个输入x，分两条线走，一条线还是x本身，另一条线是x经过11卷积，33卷积，11卷积（这三个卷积层的组合又称作bottleneck），然后把这两条线的输出做一个element-wise addition，也就是对应值相加，就是（a）中的加号，得到的结果又变成下一个同样模块的输入，几个这样的模块组合在一起就成了一个stage（比如Table1中的conv3）。（b）表示DenseNet的核心内容。（b）的左边竖着的多边形框表示输入输出内容，对输入x，只走一条线，那就是经过几层卷积后和x做一个通道的合并（cancat），得到的结果又成了下一个小模块的输入，这样每一个小模块的输入都在不断累加，举个例子：第二个小模块的输入包含第一个小模块的输出和第一个小模块的输入，以此类推。

![img](https://ai-studio-static-online.cdn.bcebos.com/8528541dfbfb4a97a1755c38baf435c4209f7b23aab9433fb23e7ca690ccd6a6)

DPN是怎么做呢？简单讲就是将Residual Network 和 Densely Connected Network融合在一起。下图中的（d）和（e）是一个意思，所以就按（e）来讲吧。（e）中竖着的矩形框和多边形框的含义和前面一样。具体在代码中，对于一个输入x（分两种情况：一种是如果x是整个网络第一个卷积层的输出或者某个stage的输出，会对x做一个卷积，然后做slice，也就是将输出按照channel分成两部分：datao1和datao2，可以理解为（e）中竖着的矩形框和多边形框；另一种是在stage内部的某个sub-stage的输出，输出本身就包含两部分：datao1和datao2），走两条线，一条线是保持datao1和datao2本身，和ResNet类似；另一条线是对x做11卷积，33卷积，11卷积，然后再做slice得到两部分c1和c2，最后c1和datao1做相加（element-wise addition）得到sum，类似ResNet中的操作；c2和datao2做通道合并（concat）得到dense（这样下一层就可以得到这一层的输出和这一层的输入），也就是最后返回两个值：sum和dense。以上这个过程就是DPN中 一个stage中的一个sub-stage。有两个细节，一个是33的卷积采用的是group操作，类似ResNeXt，另一个是在每个sub-stage的首尾都会对dense部分做一个通道的加宽操作。

![img](https://ai-studio-static-online.cdn.bcebos.com/8ba869bd3a144288b473c084c11a1c967944a9c5b18244c2b5b0cd817f613b08)

# 对读入的图像进行预处理

In [2]

```
# 对读入的图像进行预处理
def transform_img(img):
    img = cv2.resize(img, (224, 224))
    # 读入图像的数据格式是[H,W,C]
    # 使用转置操作使其变成[C,H,W]
    img = np.transpose(img, (2, 0, 1)).astype('float32')
    # 将数据范围调整到[-1.0, 1.0]之间
    img = img/255.
    img = img*2.0 - 1.0
    return img
```

# 定义训练集数据读取器

In [3]

```
def data_loader(datadir, batchsize=10, mode='train'):
    filenames = os.listdir(datadir)
    def reader():
        if mode == 'train':
            # 训练时数据乱序
            random.shuffle(filenames)
        batch_imgs = []
        batch_labels = []
        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            img = transform_img(img)

            if name[0] == 'H' or name[0] == 'N':
                # H开头的文件名表示高度近似，N开头的文件名表示正常视力
                # 高度近视和正常视力的样本，都不是病理性的，属于负样本，标签为0
                label = 0
            elif name[0] == 'P':
                # P开头的是病理性近视，属于正样本，标签为1
                label = 1
            else:
                raise ('Not support file name')

            label = np.reshape(label, [1])

            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batchsize:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array
    return reader
```

# 定义验证集数据读取器

In [4]

```
def valid_data_loader(datadir, csvfile, batch_size=10, mode='valid'):
    # 训练集读取时通过文件名来确定样本标签，验证集则通过csvfile来读取每个图片对应的标签
    # 请查看解压后的验证集标签数据，观察csvfile文件里面所包含的内容
    # csvfile文件所包含的内容格式如下，每一行代表一个样本，
    # 其中第一列是图片id，第二列是文件名，第三列是图片标签，
    # 第四列和第五列是Fovea的坐标，与分类任务无关
    # ID,imgName,Label,Fovea_X,Fovea_Y
    # 1,V0001.jpg,0,1157.74,1019.87
    # 2,V0002.jpg,1,1285.82,1080.47

    # 打开包含验证集标签的csvfile，并读入其中的内容
    file_lists = open(csvfile).readlines()

    def reader():
        batch_imgs = []
        batch_labels = []
        for line in file_lists[1:]:
            line = line.strip().split(',')
            name = line[1]
            # print(line)
            label = int(line[2])
            label = np.reshape(label, [1])
            # 根据图片文件名加载图片，并对图像数据作预处理
            file_path = os.path.join(datadir, name)
            img = cv2.imread(file_path)
            img = transform_img(img)
            # 每读取一个样本的数据，就将其放入数据列表中
            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                # 清空数据读取列表
                batch_imgs = []
                batch_labels = []


        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array
    return reader
```

# 定义DPN模型

In [5]

```
import cv2
import random
import numpy as np
import os
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear, Dropout, BatchNorm2D, Softmax
import paddle.nn.functional as F
from visualdl import LogWriter
# 设置日志保存路径
log_writer = LogWriter("./work/log")

DATADIR = 'work/palm/PALM-Training400/PALM-Training400'
DATADIR2 = 'work/palm/PALM-Validation400'
CSVFILE = 'work/labels.csv'

class DPN(paddle.nn.Layer):
    def __init__(self, num_classes=1):
        super(DPN, self).__init__()

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个卷积块，包含两个卷积
        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        # 定义第二个卷积块，包含两个卷积
        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        # 定义第三个卷积块，包含三个卷积
        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        # 定义第四个卷积块，包含三个卷积
        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        # 定义第五个卷积块，包含三个卷积
        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成线性结构(fc+relu)
        # 当输入为224*224时，经过5个卷积块和池化层后，形状变为512*7*7
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(512*7*7, 4096),
            paddle.nn.ReLU()
        )
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential将全连接层和relu组成一个线性结构（fc+relu）
        self.fc2 = paddle.nn.Sequential(
            paddle.nn.Linear(4096, 4096),
            paddle.nn.ReLU()
        )
        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, num_classes)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

        self.softmax = Softmax()



    def forward(self, x, label=None):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)

        # x = self.softmax(x)

        if label is not None:
            # print(x)
            # print(label)
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 定义训练过程
def train_pm(model, optimizer):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print("start training...")
    model.train()

    epoch_num = 10
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = data_loader(DATADIR, batchsize=10, mode='train')
    valid_loader = valid_data_loader(DATADIR2, CSVFILE)
    iter = 0
    iters = []
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)

            # 前向计算
            logits, acc = model(img, label)
            loss = F.cross_entropy(logits, label)

            avg_loss = paddle.mean(loss)
            if batch_id % 10 == 0:
                # 使用visual DL进行绘图
                iters.append(iter)
                log_writer.add_scalar(tag='acc', step=iter, value=acc.numpy())
                log_writer.add_scalar(tag='loss', step=iter, value=avg_loss.numpy())
                print('epoch:{}, batch_id:{}, loss is:{}'.format(epoch, batch_id, avg_loss.numpy()))
            iter += 1

            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()


        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            logits, acc = model(img, label)


            loss = F.cross_entropy(input=logits, label=label)
            avg_val_loss = paddle.mean(loss)
            accuracies.append(float(acc.numpy()))
            losses.append(float(avg_val_loss.numpy()))

        # 计算多个batch的平均损失和准确率
        acc_val_mean = np.array(accuracies).mean()
        avg_loss_val_mean = np.array(losses).mean()

        log_writer.add_scalar(tag='eval_acc', step=iter, value=acc_val_mean)

        print("loss={}, acc={}".format(avg_loss_val_mean, acc_val_mean))
        model.train()

    paddle.save(model.state_dict(), './work/DPN.pdparams')
```

# 创建DPN模型并训练

In [6]

```
# 创建模型
model = DPN(num_classes=2)
# 启动训练过程
# opt = paddle.optimizer.Momentum(learning_rate=0.001, momentum=0.9, parameters=model.parameters())
opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
train_pm(model, optimizer=opt)
start training...
epoch:0, batch_id:0, loss is:[1.8965338]
epoch:0, batch_id:10, loss is:[0.6905712]
epoch:0, batch_id:20, loss is:[0.7180961]
epoch:0, batch_id:30, loss is:[0.69357693]
loss=0.6922637373209, acc=0.5475000083446503
epoch:1, batch_id:0, loss is:[0.69504195]
epoch:1, batch_id:10, loss is:[0.6890862]
epoch:1, batch_id:20, loss is:[0.6810905]
epoch:1, batch_id:30, loss is:[0.6403779]
loss=0.6937279969453811, acc=0.5275000058114528
epoch:2, batch_id:0, loss is:[0.6218564]
epoch:2, batch_id:10, loss is:[0.61593544]
epoch:2, batch_id:20, loss is:[0.70049644]
epoch:2, batch_id:30, loss is:[0.69729614]
loss=0.6909015744924545, acc=0.5250000059604645
epoch:3, batch_id:0, loss is:[0.70246863]
epoch:3, batch_id:10, loss is:[0.7304039]
epoch:3, batch_id:20, loss is:[0.6946981]
epoch:3, batch_id:30, loss is:[0.7311088]
loss=0.6960067585110664, acc=0.5275000058114528
epoch:4, batch_id:0, loss is:[0.8308561]
epoch:4, batch_id:10, loss is:[0.7285439]
epoch:4, batch_id:20, loss is:[0.6756887]
epoch:4, batch_id:30, loss is:[0.6834167]
loss=0.6912791281938553, acc=0.5275000058114528
epoch:5, batch_id:0, loss is:[0.67688364]
epoch:5, batch_id:10, loss is:[0.6680447]
epoch:5, batch_id:20, loss is:[0.6894506]
epoch:5, batch_id:30, loss is:[0.6858481]
loss=0.692358547449112, acc=0.5250000059604645
epoch:6, batch_id:0, loss is:[0.6740011]
epoch:6, batch_id:10, loss is:[0.6869786]
epoch:6, batch_id:20, loss is:[0.68561965]
epoch:6, batch_id:30, loss is:[0.7218032]
loss=0.6904958054423332, acc=0.5275000058114528
epoch:7, batch_id:0, loss is:[0.6810826]
epoch:7, batch_id:10, loss is:[0.68289745]
epoch:7, batch_id:20, loss is:[0.68964267]
epoch:7, batch_id:30, loss is:[0.6794151]
loss=0.6917793869972229, acc=0.5275000058114528
epoch:8, batch_id:0, loss is:[0.72442955]
epoch:8, batch_id:10, loss is:[0.7193705]
epoch:8, batch_id:20, loss is:[0.69215643]
epoch:8, batch_id:30, loss is:[0.68562275]
loss=0.6922665894031524, acc=0.5275000058114528
epoch:9, batch_id:0, loss is:[0.66816545]
epoch:9, batch_id:10, loss is:[0.6579969]
epoch:9, batch_id:20, loss is:[0.6806724]
epoch:9, batch_id:30, loss is:[0.729876]
loss=0.691317941248417, acc=0.5275000058114528
```

In [ ]

```
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```