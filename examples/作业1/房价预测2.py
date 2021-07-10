import torch
import torch.nn as nn # 搭建神经网络
from torch.optim import SGD # 优化器SGD
import torch.utils.data as Data # 数据预处理

from sklearn.datasets import load_boston # 导入波士顿房价数据
from sklearn.preprocessing import StandardScaler # 数据标准化

# import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


#%%
# (1) Data prepare 数据准备

# 读入数据
boston_X, boston_y = load_boston(return_X_y=True)
print("boston_X.shape: ", boston_X.shape)

plt.figure()
plt.hist(boston_y, bins=20) # hist方法来绘制直方图
plt.show()

# 标准化数据
ss = StandardScaler(copy=True, with_mean=True, with_std=True)
boston_Xs = ss.fit_transform(boston_X)


# 使用GPU计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将数据转化为张量
train_xt = torch.from_numpy(boston_Xs.astype(np.float32)).cuda(device)
print(train_xt.size()) # 数据尺寸为(506, 13), 其中13为波士顿房价数据的特征数,
                       # 13将作神经网络的输入层神经单元个数
                       
train_yt = torch.from_numpy(boston_y.astype(np.float32)).cuda(device)
print(train_yt.size()) 


# 制作训练用数据，即关联数据，将输房价特征数据train_xt与房价数据train_y关联train_data = Data.TensorDataset(train_xt, train_yt)
print(train_data[0: 5]) # 预览前5组数据 

# 定义一个数据加载器，用于批量加载训练用数据, 让数据分批次进入神经网络
train_loader = Data.DataLoader(dataset=train_data, 
                               batch_size=128,
                               shuffle=True,
                               num_workers=0)

#%%
# (2) 定义网络模型
# 以类的方式继承基类,定义个多层感知机
class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        # First hidden layer
        self.h1 = nn.Linear(in_features = 13, out_features=30, bias=True)
        self.a1 = nn.ReLU()
        # Second hidden layer
        self.h2 = nn.Linear(in_features=30, out_features=10)
        self.a2 = nn.ReLU()
        # regression predict layer
        self.regression = nn.Linear(in_features=10, out_features=1)
        
    def forward(self, x):
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        output = self.regression(x)
        return output
  
mlp_1 = MLPmodel().cuda()  #网络转到GPU上计算
list(mlp_1.parameters())[0].device   # 查看网络是否到GPU上
print(mlp_1)


#%%
# (3) 训练
optimizer = SGD(mlp_1.parameters(), lr=0.001) # 定义优化器 define Optimizer
loss_function = nn.MSELoss()    # 定义损失函数loss function

train_loss_all = [] # 存放每次迭代的误差数据，便于可视化训练过程

# Train
for epoch in range(60): # 迭代总轮数：50次
    # 对每个批次进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlp_1(b_x).flatten()
        train_loss = loss_function(output, b_y) # 误差计算
        optimizer.zero_grad() # 梯度置位，或称梯度清零
        train_loss.backward() # 反向传播，计算梯度
        optimizer.step() # 梯度优化
        train_loss_all.append(train_loss.item())
    print("train epoch %d, loss %s:" % (epoch + 1, train_loss.item()))

# 可视化训练过程（非动态）
plt.figure()
plt.plot(train_loss_all, "g-")
plt.title("MLPmodel1: Train loss per iteration")
plt.show()
        

#%%
# 使用nn.Sequential()创建网络模型 , 并增加隐藏层的层数和每层的神经单元个数
class MLPmodel2(nn.Module):
    def __init__(self):
        super(MLPmodel2, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(in_features=13, out_features=40, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=40, out_features=30, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=30, out_features=10, bias=True),
            nn.ReLU(),
            )
        self.regression = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output

mlp_2 = MLPmodel2().cuda() # 网络转到GPU上计算
list(mlp_2.parameters())[0].device  
print(mlp_2)

# 训练
optimizer = SGD(mlp_2.parameters(), lr=1e-3) # 定义优化器 define Optimizer
loss_function = nn.MSELoss()    # 定义损失函数loss function

train_loss_all = [] # 存放每次迭代的误差数据，便于可视化训练过程

# Train
for epoch in range(60): # 迭代总轮数：50次
    # 对每个批次进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlp_2(b_x).flatten()
        train_loss = loss_function(output, b_y) # 误差计算
        optimizer.zero_grad() # 梯度置位，或称梯度清零
        train_loss.backward() # 反向传播，计算梯度
        optimizer.step() # 梯度优化
        train_loss_all.append(train_loss.item())
    print("train epoch %d, loss %s:" % (epoch + 1, train_loss.item()))

# 可视化训练过程（非动态）
plt.figure()
plt.plot(train_loss_all, "g-")
plt.title("MLPmodel2: Train loss per iteration")
plt.show()
        