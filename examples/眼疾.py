#!/usr/bin/env python
# coding: utf-8

# In[1]:


#picture->path train_data
def init_process(path, lens):
    data = []
    name = find_label(path)
    for i in range(lens[0], lens[1]+1):
        data.append([path % i, name])
        
    return data


# In[2]:


def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i] == '%' :
            last = i - 1
        if (str[i] == 'H' or str[i] == 'N' or str[i] == 'P') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last+1]
    if name == 'H' or name == 'N':
        return 0 #正常
    else:
        return 1 #异常


# In[3]:


import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import os


# In[4]:


def Myloader(path):
    return Image.open(path).convert('RGB')


# In[5]:


class MyDataset(Dataset):#重写dataset类
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


# In[6]:


DATADIR2='C:/Users/Lhh/pycharmProjects/pythonProject/data2/Training400/validation400/'
CSVFILE = 'C:/Users/Lhh/pycharmProjects/pythonProject/data2/Validation-GT/labels.csv'
filelists = open(CSVFILE).readlines()
testimgs = []
for line in filelists[1:]:
    line = line.strip().split(',')
    name = line[1]
    label = int(line[2])
    # 存放验证集的路径及结果
    filepath = os.path.join(DATADIR2, name)
    testimgs.append([filepath,label])


# In[7]:


def load_data():
    transform = transforms.Compose([
        transforms.CenterCrop(299),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    
    path1 = 'C:/Users/Lhh/pycharmProjects/pythonProject/data2/Training400/Training400/H%d.jpg'
    data1 = init_process(path1, [1, 26])
    path2 = 'C:/Users/Lhh/pycharmProjects/pythonProject/data2/Training400/Training400/N%d.jpg'
    data2 = init_process(path2, [1, 161])
    path3 = 'C:/Users/Lhh/pycharmProjects/pythonProject/data2/Training400/Training400/P%d.jpg'
    data3 = init_process(path3, [1, 213])

    train_data = data1 + data2 + data3

    train = MyDataset(train_data, transform=transform, loder=Myloader)

    test_data = testimgs
    test= MyDataset(test_data, transform=transform, loder=Myloader)

    train_data = DataLoader(dataset=train, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    test_data = DataLoader(dataset=test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    return train_data, test_data


# In[8]:


import torchvision.models as models
model=models.inception_v3(pretrained=True)
model.fc=torch.nn.Sequential(torch.nn.Linear(2048,2,bias=True))
model = model.cuda()


# In[11]:


train_loader, test_loader = load_data()


# In[12]:


n_epochs=10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=0.0001
optimizer = optim.Adam(model.parameters())
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)#lr随着训练不断衰减
criterion = nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    running_loss=0.0
    correct=0
    print("Epoch {}/{}".format(epoch+1,n_epochs))
    print("-"*10)
    for batch_idx, (data, target) in enumerate(train_loader, 0):
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        optimizer.zero_grad()  # 梯度清0
        outputs = model(data)[0]  # 取0即不考虑辅助分类器的结果
        pred = torch.max(outputs.data, 1)[1].data
        loss = criterion(outputs, target)  # 计算误差
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss+=loss.data
        correct += (pred== target).sum()
        
    current=0
    total=0
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
        
    print("Loss:{:.4f},Train Accuracy:{:.2f}%,Valid Accuracy:{:.2f}%".format(running_loss,100*correct/400,100*current/total))

