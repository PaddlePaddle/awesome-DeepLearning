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


# In[5]:


import numpy as np
import pandas as pd


# In[6]:


path = '/home/aistudio/data/data5646/housing.csv'
data = pd.read_csv(path, header=0)
data = np.array(data)


# In[7]:


ratio = 0.8
train_num  =int(ratio*data.shape[0])
train_data = data[:train_num]
test_data = data[train_num:, ]
maxn, minn, avgs = np.max(train_data, axis=0) , np.min(train_data, axis=0), np.sum(train_data, axis=0) / train_data.shape[0]
feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO','B', 'LSTAT', 'MEDV']
for i in range(len(feature_names)):
    data[:, i] = (data[:, i] - avgs[i])/( maxn[i] - minn[i])
'''
var = np.var(train_data, axis=0)
avgs = np.sum(train_data, axis=0) / train_data.shape[0]
for i in range(len(feature_names)):
    data[:, i] = ((data[:, i]) - avgs[i])  / var[i] 
'''


# 以上就是对数据的导入以及预处理（归一化和标准化），可将其组合起来封装成一个函数。

# In[8]:


class Network():

    def __init__(self, num):
        np.random.seed(0)
        self.w1 = np.random.random([num, 5])   #没有先验知识或者数学指导，不知道中间层的节点该定义为多少，随便定义了一个20
        self.b1 = np.zeros([1, 5])
        self.w2 = np.random.random([5, 1])
        self.b2 = 0
        self.gradient1 = 0
        self.gradient2 = 0
        self.g1 = 0
        self.g2 = 0
    
    def forward(self, x):                     #relu
        z1 = np.dot(x, self.w1) + self.b1
        bck = z1
        for i in range (5):
            if z1[0, i] <= 0 :
                bck[0, i] =0
        z2 = np.dot(bck, self.w2) + self.b2 
        return z2

    def loss(self, x, y):
        err  = x-y
        cost = err*err
        cost = np.mean(cost) / 2
        return cost
    
    def bp(self, x ,y): #对所有样本的BP算法,这里要求样本要一个一个输入，两层之间的激活函数是Relu。
        lr = 0.1
        z1 = np.dot(x, self.w1)+ self.b1 #第一层的输出
        z2 = self.forward(x) #第二层的输出
        g = (y-z2) #中间变量g
        self.g1 -= g 
        self.gradient1 += lr*g*z1   #第二层的梯度
        e = self.w2*g
        for i in range (5):
            if z1[0, i] <= 0:
                e[i, 0] = 0
        self.g2 -= e 
        x = np.reshape(x, (13, 1))
        self.gradient2 += lr*np.dot(x, np.transpose(e))  #第一层的梯度,对每一个数据产生的梯度累加求和


    
    def mn(self, num):
        self.gradient1 = self.gradient1 / num
        self.gradient2 = self.gradient2 / num  #对梯度需要取平均值否则会梯度爆炸
        self.g1 = self.g1 / num
        self.g2 = self.g2 /num


    def update(self):
        self.w1 += self.gradient2
        self.w2 += np.transpose(self.gradient1)
        self.b1 -= np.transpose(self.g2)
        self.b2 -= self.g1

    def grad_zero(self):
        self.gradient1, self.gradient2, self.g1, self.g2 = 0, 0, 0, 0
        
    #梯度更新及清零
# 两层神经网络的定义


# In[13]:


net = Network(13)
x = train_data[:, :13]
y = train_data[:, -1]
y = np.reshape(y, (404, 1))


for i in range(2000):
    for j in range(train_data.shape[0]):
        net.bp(x[j], y[j])    
        net.mn(train_data.shape[0])
        net.update()
        net.grad_zero()
print(net.w1)


# In[20]:


def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])

    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label

one_data, label = load_one_example()

predict = net.forward(one_data)

# 对结果做反归一化处理
predict = predict * (maxn[-1] - minn[-1]) + avgs[-1]
# 对label数据做反归一化处理
label = label * (maxn[-1] - minn[-1]) + avgs[-1]

print("Inference result is {}, the corresponding label is {}".format(predict, label))


# 对于两个网络的比价的说明相关部分在神经网络的代码文件里
