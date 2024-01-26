#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys 
sys.path.append('/home/aistudio/external-libraries')
from paddle.vision.transforms import Compose, Normalize
import paddle
import paddle.nn.functional as F
import numpy as np
from paddle.metric import Accuracy
import random
from paddle import fluid
from visualdl import LogWriter


# In[6]:


log_writer=LogWriter("./data/log/train") #log记录器


transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])


# In[ ]:


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

