#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[3]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[4]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[2]:


import pandas as pd
import paddle
import numpy as np
from paddle.nn import Linear
import paddle.nn.functional as F
#导入必要的库


# In[3]:


path = '/home/aistudio/data/data5646/housing.csv'
data = pd.read_csv(path, header=0)
data = np.array(data)

ratio = 0.8
train_num  =int(ratio*data.shape[0])
train_data = data[:train_num]
test_data = data[train_num:, ]
maxn, minn, avgs = np.max(train_data, axis=0) , np.min(train_data, axis=0), np.sum(train_data, axis=0) / train_data.shape[0]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B','LSTAT', 'MEDV']
for i in range(len(feature_names)):
    data[:, i] = (data[:, i] - avgs[i])/( maxn[i] - minn[i]) #归一化
'''
var = np.var(train_data, axis=0)
avgs = np.sum(train_data, axis=0) / train_data.shape[0]
for i in range(len(feature_names)):
    data[:, i] = ((data[:, i]) - avgs[i])  / var[i] #标准化
 '''


# In[4]:


class Network(paddle.nn.Layer):
    
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = Linear(in_features=13, out_features=20)
        self.fc2 = Linear(in_features=20, out_features=1)
    
    def forward(self, x):
        z1 = self.fc1(x)
        z2 = F.relu(z1)
        z3 = self.fc2(z2)
        return z3
#两层网络的结构


# In[5]:


model = Network()
model.train()
opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
#常规设置


# In[6]:


epoch = 2000
batch_size = 10
l = 10000
for i in range (epoch):
    np.random.shuffle(train_data)
    mini_batches = [train_data[k:k+batch_size] for k in range(0, len(train_data), batch_size)] #打乱 拆分数据集
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])
        x = np.float32(x)
        y = np.array(mini_batch[:, -1])
        y = np.float32(y)
        features = paddle.to_tensor(x)
        price = paddle.to_tensor(y)

        pred = model(features)
        loss = F.square_error_cost(pred, price)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        opt.step()
        opt.clear_grad()
    if i % 200 ==0 :
        print(avg_loss)

    #保存loss最小,即最优的模型参数
       
        if avg_loss < l :
            l = avg_loss
            paddle.save(model.state_dict(), 'LR_model.pdparams')
            print("模型保存成功，模型参数保存在LR_model.pdparams中")


# In[11]:


def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label


# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data, label = load_one_example()
# 将数据转为动态图的variable格式 
one_data = np.float32(one_data)
one_data = paddle.to_tensor(one_data)
predict = model(one_data)

# 对结果做反归一化处理
predict = predict * (maxn[-1] - minn[-1]) + avgs[-1]
# 对label数据做反归一化处理
label = label * (maxn[-1] - minn[-1]) + avgs[-1]

print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))


# 网络结构为全连接+Relu+全连接的两层神经网络分别用Numpy和paddlepaddle成功编写。实现了线性回归的功能，能够比较准确的预测房价。
# 在损失函数这一点，问题的本质都是线性回归，对应的误差的概率分布是一样的，优化函数也就是损失函数也就是一样的，都是均方差损失MSELoss，可能写的形式不同。
# 在经过训练2000次之后，经过多次取值实验，大部分时候预测值和真实值的误差都在2以内，误差大的时候也一般不超过10，并且两个模型在此时性能接近，用paddlepaddle编写的网络要略优一点。
# 
# 
# 
