#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入需要用到的package
import numpy as np
import matplotlib.pyplot as plt
import json
# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')
data


# In[2]:


# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])


# In[3]:


# 查看数据
x = data[0]
print(x.shape)
print(x)


# In[4]:


ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
training_data.shape


# In[5]:


# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs =                      training_data.max(axis=0),                      training_data.min(axis=0),      training_data.sum(axis=0) / training_data.shape[0]
# 对数据进行归一化处理
for i in range(feature_num):
    #print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])


# In[6]:


def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',                       'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0),                                  training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# In[7]:


# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[8]:




class Network(object):
    #def __init__(self, num_of_weights):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
       #np.random.seed(0)
       self.w0 = np.random.randn(num_of_weights, num_of_weights)
       self.w1 = np.random.randn(num_of_weights, 1)
       self.b0 = 0.
       self.b1 = 0.
        
    def forward(self, x):
        z1 = np.dot(x, self.w0) + self.b0
        z2=np.maximum(z1,0)
        z3 =np.dot(z2,self.w1) + self.b1
        return z1,z2,z3
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    #梯度更新参数
    def gradient(self, x, y):
        z1,z2,z3 = self.forward(x)
        N = x.shape[0]
        gradient_w0 = 1. / N * np.sum((z3 - y) * z2 * x, axis=0)
        gradient_w0 = gradient_w0[:, np.newaxis]
        gradient_b0 = 1. / N * np.sum((z3 - y)*(y-z2))
        gradient_w1 = 1. / N * np.sum((z3 - y) * z2, axis=0)
        gradient_w1 = gradient_w1[:, np.newaxis]
        gradient_b1 = 1. / N * np.sum(z3 - y)
        return gradient_w0, gradient_b0,gradient_w1,gradient_b1


    def update(self, gradient_w0, gradient_b0,gradient_w1,gradient_b1, eta = 0.01):
        self.w0 = self.w0 - eta * gradient_w0
        self.b0 = self.b0 - eta * gradient_b0
        self.w1 = self.w1 - eta * gradient_w1
        self.b1 = self.b1 - eta * gradient_b1    
                
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z1,z2,z3= self.forward(x)
                loss = self.loss(z3, y)
                gradient_w0,gradient_b0,gradient_w1,gradient_b1 = self.gradient(x, y)
                self.update(gradient_w0,gradient_b0,gradient_w1,gradient_b1, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses

# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()



    
  


# In[ ]:





# In[ ]:




