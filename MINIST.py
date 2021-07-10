#加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

# 设置数据读取器，API自动读取MNIST数据训练集
train_dataset = paddle.vision.datasets.MNIST(mode='train')

# 导入需要的包
import paddle
import paddle.fluid as fluid
import numpy as np
#from paddle.fluid.dygraph import Conv2D, Pool2D, Linear, Dropout, BatchNorm

class InceptionA(fluid.dygraph.Layer):
    def __init__(self,  in_channels):
        super(InceptionA, self).__init__()
        self.branch3x3_1 = paddle.nn.Conv2D(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = paddle.nn.Conv2D(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = paddle.nn.Conv2D(24, 24, kernel_size=3, padding=1)

        self.branch5x5_1 = paddle.nn.Conv2D(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = paddle.nn.Conv2D(16, 24, kernel_size=5, padding=2)

        self.branch1x1 = paddle.nn.Conv2D(in_channels, 16, kernel_size=1)
        self.branch_pool = paddle.nn.Conv2D(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch1x1 = self.branch1x1(x)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return fluid.layers.concat(outputs,axis=1)#24+24+16+24=88个通道


# 定义mnist数据识别网络结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义一层全连接层，输出维度是1
        self.conv1 = paddle.nn.Conv2D(1, 10, kernel_size=5)
        self.conv2 = paddle.nn.Conv2D(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = paddle.nn.MaxPool2D(2)
        self.fc = paddle.nn.Linear(1408, 10)

    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = F.relu(self.mp(self.conv1(inputs)))
        outputs = self.incep1(outputs)
        outputs = F.relu(self.mp(self.conv2(outputs)))
        outputs = self.incep2(outputs)
        outputs = paddle.flatten(outputs, start_axis=1, stop_axis=3)
        outputs = self.fc(outputs)
        return outputs

# 声明网络结构
model = MNIST()

def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())


# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 256
    # 将图像形式reshape为[batch_size, 784]
    # img = paddle.reshape(img, [batch_size, img_h*img_w])

    return img


import paddle

# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')


def train(model):
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 5
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = np.reshape(norm_img(data[0]).astype('float32'), (16, 1, 28, 28))
            # print(images.shape)#--------16*28*28
            images = paddle.to_tensor(images)
            labels = data[1].astype('float32')
            labels = paddle.to_tensor(labels, 'int64')

            # 前向计算的过程
            predicts = model(images)

            # 计算损失
            loss = F.cross_entropy(predicts, labels)

            avg_loss = paddle.mean(loss)

            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()




train(model)
paddle.save(model.state_dict(), './mnist.pdparams')

from paddle.metric import Accuracy
test_dataset = paddle.vision.datasets.MNIST(mode='test')
#随机抽取图片进行测试
def random_test(model,num=16):
    test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=16,shuffle=True)
    for test_batch_id, test_data in enumerate(test_loader()):
        x_data = np.reshape(norm_img(test_data[0]).astype('float32'),(16,1,28,28))
        x_data=paddle.to_tensor(x_data)
        label = paddle.to_tensor(test_data[1],'int64')
    predicts = model(x_data)
    #返回正确率
    acc = paddle.metric.accuracy(predicts, label)
    print("The accuracy tested in randomly elected samples ：{}".format(acc.numpy()))

random_test(model,num=16)


