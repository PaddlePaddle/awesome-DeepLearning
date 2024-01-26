import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torchvision

transform = transforms.Compose([       
    transforms.ToTensor(),        
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))   #将每个元素分布到（-1，1），由于彩色图像为三通道所以均值和方差均为1*3
])

batch_size = 64 

#设置参数
batch_size_train = 64#训练集每批次加载64张图片
batch_size_test = 1000#测试集每批次加载1000张图片
learning_rate = 0.01#学习率，默认为0.01
momentum = 0.5#动量参数，加入动量参数（1）不仅考虑了当前的梯度下降的方向，还综合考虑了上次更新的情况，使得学习器学习过程的幅度和角度不会太尖锐，特别是遇到来回抖动的情况时大有好转。
#（2）当遇到局部极小值时，因为设置了惯性，所以可能可以冲出局部最小值，这样就更可能找到一个全局最小值。
log_interval = 10#运行10批次打印一次结果
#准备数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data',train=True,download=False,
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data',train=False,download=False,
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
print(len(test_loader))
print(len(train_loader))




class InceptionA(nn.Module):     #定义inception模块
    def __init__(self, in_channels):       #定义网络的四条支路，结构并联
        super(InceptionA, self).__init__()
        self.submodel1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3,  padding=1),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)
        )
        self.submodel2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5,  padding=2)
        )
        self.submodel3 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
            
        self.branch_pool=nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1) 

    def forward(self, x):     #定义网络的前向传播
        x1 = self.submodel1(x)      #每一个分支的输出
        x2 = self.submodel2(x)
        x3 = self.submodel3(x)
        x4 = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        x4 = self.branch_pool(x4)

        return torch.cat([x1, x2, x3, x4], dim=1)       #返回四个分支的结果的堆叠


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Conv2d(1,10,5)
        self.conv2=nn.Conv2d(88,20,5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(1408,10)

    def forward(self, x):   
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)       
        return x

# --------------------------------------网络实例化---------------------------------------------------------------------- #
model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)   #学习率为0.01  动量为0.5
loss_function = nn.CrossEntropyLoss()    #交叉熵损失函数

# -----------------------------训练------------------------------------------------------------------------- #
epoch_size = 5

for epoch in range(epoch_size):
    print('epoch',epoch+1)
    model.train()
    train_acc = 0             #参数初始化
    train_loss = 0
    test_acc = 0
    test_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data
        label = label

        out = model(data)

        loss = loss_function(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        prediction = torch.argmax(out, dim=1)
        train_acc += torch.sum(prediction.eq(label)).item()     #用于计算训练集准确率

    with torch.no_grad():
        model.eval()
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data
            label = label

            out = model(data)
            loss = loss_function(out, label)
            prediction = torch.argmax(out, dim=1)
            test_acc += torch.sum(prediction.eq(label)).item()      #用于计算测试集准确率
            test_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)
    test_acc /= len(test_loader)
    test_loss /= len(test_loader)

    
    print('TRAIN:   accurary:{}   loss:{}'.format(train_acc, train_loss))
    print('TEST :   accurary:{}   loss:{}'.format(test_acc, test_loss))

torch.save(model, './model6.pkl')     #模型保存
print('saved')





















