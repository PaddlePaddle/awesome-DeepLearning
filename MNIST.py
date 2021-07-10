import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import os
from torchvision import datasets, transforms

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
lr = 0.01
momentum = 0.5
log_interval = 10
epochs = 10
test_batch_size = 1000
batch_size = 64
num_classes = 10
num_epochs = 10
#60000个训练数据 10000个测试数据
train_loader = torch.utils.data.DataLoader(datasets.MNIST(root = './data',train=True,download=True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
                batch_size = batch_size,shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST(root = './data',train=False,download=True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
                batch_size = test_batch_size,shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()     
        self.conv1 = nn.Sequential(     #输入1*28*28
            nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size=5, stride=1, padding=2),    #2保证输入输出尺寸相同
            nn.BatchNorm2d(6),
            nn.ReLU(),          #输入6*28*28
            nn.MaxPool2d(kernel_size=2,stride=2)  #输出6*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6,16,kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),           #输入16*10*10
            nn.MaxPool2d(kernel_size=2,stride=2)    #输出16*5*5
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),   #输出120*1*1
            nn.BatchNorm2d(120),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(84, num_classes),
            nn.LogSoftmax()
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.reshape(out.size(0), -1)   #nn.Linear()输入输出都是一维的值，所以要把多维度tensor展平成1维
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train(epoch):
    model.train()   #设置为trainning模式
    for batch_idx,(data,target) in enumerate(train_loader):
        data = torch.tensor(data).type(torch.FloatTensor).to(device)
        target = torch.tensor(target).type(torch.LongTensor).to(device)
        
        #前向传播
        output = model(data)
        loss = nn.CrossEntropyLoss()(output,target)  #交叉熵作为损失函数

        #反向传播和优化
        torch.optim.SGD(model.parameters(),lr = lr,momentum = momentum).zero_grad()  #SGD算法优化
        loss.backward()
        torch.optim.SGD(model.parameters(),lr =lr,momentum = momentum).step()   #结束一次前传+反传之后，更新参数

        if batch_idx % log_interval == 0 :
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch,batch_idx*len(data),len(train_loader.dataset),100.*batch_idx/len(train_loader),loss.item()))

def test():
    model.eval()
    test_loss = 0   #初始化测试损失值为0
    correct = 0   #初始化预测正确的数据个数为0
    for data,target in test_loader:
        data = torch.tensor(data).type(torch.FloatTensor).to(device)
        target = torch.tensor(target).type(torch.LongTensor).to(device)
        output = model(data)

        test_loss += nn.CrossEntropyLoss()(output.data,target).item()   #loss值累加

        _,predicted = torch.max(output.data,1)   #获得最大概率的下标
        correct += (predicted == target).sum().item()   #预测正确的个数累加

    test_loss = test_loss / len(test_loader.dataset)  # 得到平均loss
    print('\nTest set:Average loss:{:.8f},Accuracy:{}/{}({:.2f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))

if __name__ == '__main__':
    model = LeNet().to(device)
    for epoch in range(num_epochs):
        train(epoch)
    print("Train Finish!")
    test()