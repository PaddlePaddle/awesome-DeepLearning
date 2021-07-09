#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

data = load_boston()
X = data['data']  # 这样就把特征提取出来了
y = data['target']  # 这样就把目标target给提取出来了

y = y.reshape(-1, 1)  # y这个时候还是一维的的ndarray   需要把他转换成一个列

# 数据规范化
ss = MinMaxScaler()
X = ss.fit_transform(X)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
# 从ndarray格式转成torch格式
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.FloatTensor)

# 构造网络
model = nn.Sequential(
    nn.Linear(13, 10),  # 13*10   因为之前的下面的输入train_x的维度是(404,13)
    nn.ReLU(),  # 这里是一层ReLU()层  输出的数据的维度(404,10)
    nn.Linear(10, 1)  # 再加一层全连接层 就输出y了  维度是(404,1)
)
# 构造优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
# 训练
max_epoch = 301
iter_loss = []
for i in range(max_epoch):
    # 前向传播
    y_pred = model(train_x)  # 把train_x喂入
    # 计算loss
    loss = criterion(y_pred, train_y)  # 计算预测值跟实际的误差
    if i % 30 == 0:
        print("第{}次迭代的loss是:{}".format(i, loss))
    iter_loss.append(loss.item())
    # 清空之前的梯度
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 权重调整
    optimizer.step()

output = model(test_x)  # 模型训练好了 再把测试集给喂入  然后就得到了测试集的输出值
predict_list = output.detach().numpy()

# 绘制不同的 iteration 的loss
x = np.arange(max_epoch)
y = np.array(iter_loss)
plt.figure()
plt.plot(x, y)
plt.title("the loss of iteration step")
plt.xlabel("iteration step")
plt.ylabel("loss")
plt.show()

# 查看真实值与预测值的散点图
x = np.arange(test_x.shape[0])
y1 = np.array(predict_list)  # 测试集的预测值
y2 = np.array(test_y)  # 测试集的实际值
line1 = plt.scatter(x, y1, c="black")
line2 = plt.scatter(x, y2, c='red')
plt.legend([line1, line2], ["y_Predict", "y_True"])
plt.title("the loss between y_True and y_Predict")
plt.ylabel("the price of Boston")
plt.show()


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
