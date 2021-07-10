#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torchvision
import torch
import torch.utils.data.dataloader as Data
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


# In[ ]:


#残差块
if_use_gpu=0
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,padding=1,stride=stride,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = nn.Sequential()
        #输入输出信道数不一样，把残差块的信道卷积到和输出一样
        if(inchannel != outchannel):
            self.right = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.right(x)
        out =F.relu(out)
        return out


# In[ ]:


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=1)
        self.conv2 = nn.Conv2d(128,128,3,stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2)
        #self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=1)
        self.conv4 = nn.Conv2d(256,256,6)
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        layer = []
        for i in range(num_blocks):
            layer.append(block(self.inchannel,channels,stride))
            self.inchannel = channels
        #对layer拆包
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        #out = self.layer4(out)
        out = self.conv4(out)
        #out = F.avg_pool2d(out,4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# In[ ]:


def ResNet18():

    return ResNet(ResidualBlock)

train_data = torchvision.datasets.MNIST(
    './mnist', train=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ]), download=True
)
train_data.data = train_data.data[:10000]
train_data.targets = train_data.targets[:10000]
test_data = torchvision.datasets.MNIST(
    './mnist', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ])
)
print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=32)

model = ResNet18()
if if_use_gpu:
    model = model.cuda()

print(model)


# In[ ]:


optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
for epoch in range(1):
    print('epoch {}'.format(epoch + 1))
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        batch_x, batch_y = Variable(inputs), Variable(labels)
        if if_use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        out = model(batch_x)
        batch_y = batch_y.long()
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 返回每行元素最大值
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_correct = train_correct.item()
        train_loss = loss.item()
        print('batch:{},Train Loss: {:.6f}, Acc: {:.6f}'.format(i+1,train_loss , train_correct /32))


# In[ ]:


# evaluation--------------------------------
model.eval()
eval_loss = 0.
eval_acc = 0.
for batch_x, batch_y in test_loader:
batch_x, batch_y = Variable(batch_x, requires_grad=False), Variable(batch_y,requires_grad=False)
if if_use_gpu:
    batch_x = batch_x.cuda()
    batch_y = batch_y.cuda()
out = model(batch_x)
loss = loss_func(out, batch_y)
eval_loss += loss.item()
pred = torch.max(out, 1)[1]
num_correct = (pred == batch_y).sum()
eval_acc += num_correct.item()
print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
test_data)), eval_acc / (len(test_data))))


# In[ ]:





# In[ ]:





# In[ ]:




