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
import paddle
print(paddle.__version__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


# In[6]:


colnames = ['房屋面积']+['房价']
print_data = pd.read_csv('./房价预测/data/data.txt',names = colnames)
print_data.head()


# In[9]:


global x_raw,train_data,test_data
data = np.loadtxt('./房价预测/data/data.txt',delimiter=',')
x_raw = data.T[0].copy() 

#axis=0,表示按列计算
#data.shape[0]表示data中一共有多少列
maximums,minimums,avgs = data.max(axis=0),data.min(axis=0),data.sum(axis=0)/data.shape[0]
print("the raw area :",data[:,0].max(axis = 0))
#归一化
data[:,0] = (data[:,0]-avgs[0])/(maximums[0]-minimums[0])
print('normalization:',data[:,0].max(axis = 0))
plt.plot(data[:,0],data[:,1])
plt.show()


# In[11]:


#划分数据集
ratio = 0.8
offset = int(data.shape[0]*ratio)
train_data = data[0:offset]
test_data = data[offset:]
print('数据总个数：',len(data))
print('训练集数据个数:',len(train_data))
print('测试集数据个数:',len(test_data))


# In[12]:


import paddle
from paddle.io import Dataset

class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, Data):
        """
        步骤二：实现构造函数，定义数据集大小
        """
        super(MyDataset, self).__init__()
        self.num_samples = len(Data)
        self.Data = Data.astype('float64')

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        data = self.Data[index,0]
        label = self.Data[index,1]

        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return self.num_samples




# In[13]:


BATCH_SIZE = 1
train_loader = paddle.io.DataLoader(MyDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
test_loader = paddle.io.DataLoader(MyDataset(test_data), batch_size=BATCH_SIZE, shuffle=True)


# In[14]:


#建立模型
class net(paddle.nn.Layer):
    def __init__(self):
        super(net, self).__init__()
        self.fc = paddle.nn.Sequential(
        paddle.nn.Linear(1, 1),
        paddle.nn.ReLU(),
        paddle.nn.Linear(1, 1)
        )

    def forward(self, inputs):
        pred = self.fc(inputs)
        return pred


# In[15]:


paddle.set_default_dtype("float64")
# step3:训练模型
model = paddle.Model(net())
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.MSELoss())
model.fit(train_loader, test_loader, epochs=5, batch_size=20, verbose=1)


# ## Numpy实现

# In[28]:


# 导入需要用到的package
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 从文件导入数据
    '''
    datafile = './housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])
'''
    data = np.loadtxt('./房价预测/data/data.txt',delimiter=',')
    x_raw = data.T[0].copy() 

    #axis=0,表示按列计算
    #data.shape[0]表示data中一共有多少列
    maximums,minimums,avgs = data.max(axis=0),data.min(axis=0),data.sum(axis=0)/data.shape[0]
    print("the raw area :",data[:,0].max(axis = 0))
    #归一化
    data[:,0] = (data[:,0]-avgs[0])/(maximums[0]-minimums[0])
    print('normalization:',data[:,0].max(axis = 0))

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, loss))

        return losses


# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(1)
# 启动训练
losses = net.train(train_data, num_epoches=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

