#!/usr/bin/env python
# coding: utf-8

# ## 使用python+numpy实现房价预测

# In[13]:


## 预先导入一些第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
get_ipython().run_line_magic('matplotlib', 'inline')


# **在创建环境的时候选择导入`波士顿房价预测`数据集，数据集位于`/home/data/data7802/boston_house_price.csv`，数据集部分展示如下：**
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/d004d7ac78d440738da592d8695bb1e5c4efb9310d7e4ae483a502bf0cb53155)
# 

# In[14]:


## 读取数据集并展示一部分数据
house_price_data = pd.read_csv("data/data7802/boston_house_prices.csv", header=1)
print(house_price_data.head())


# **其中前13项都是可能影响房价的一些因素，最后一列`MEDV`表示的就是房价**

# In[15]:


## 创建一个神经网络模型
class GD(object):
    def __init__(self, x, y):
        # 数据集
        self.x = x
        self.y = y.reshape(-1, 1)
        # 默认学习率
        self.lr = 1e-3
        # 训练次数
        self.times = 500
        # 随机初始化参数，包括w和b
        self.theta = np.random.randn(self.x.shape[1] + 1, 1)
        # 在self.x后边加上一列值全为1的向量
        self.X = np.column_stack((self.x, np.ones([self.x.shape[0], 1])))
        # 保存误差
        self.losslist = []

    # 计算梯度
    def CalculateGradient(self):
        """
        计算当前梯度，并且返回
        X^T*(X*theta - y)
        """
        return np.matmul(self.X.T, np.matmul(self.X, self.theta) - self.y)

    def CalculateLoss(self):
        """
        计算每次迭代过程中的误差值,并且附加到self.losslist中
        :return: None
        """
        loss = 0.5 * np.sum(np.power(np.matmul(self.X, self.theta) - self.y, 2))
        self.losslist.append(loss)

    def GradientDeecent(self):
        """
        梯度下降，更新参数,并且返回参数
        :return: self.theta, self.b
        """
        for i in range(self.times):
            # 保存当前梯度信息
            g = self.CalculateGradient()
            # 计算误差
            self.CalculateLoss()
            # 更新参数
            self.theta = self.theta - self.lr * g
            print("训练轮次：{}，当前轮次误差：{}".format(i + 1, self.losslist[i]))


# In[17]:


## 读取数据并训练
x = house_price_data.iloc[:, 0:13]
y = house_price_data.iloc[:, 13]
x_norm = ((x - x.min()) / (x.max() - x.min())).values
y_norm = ((y - y.min()) / (y.max() - y.min())).values
# 实例化
gd = GD(x_norm, y_norm)
gd.GradientDeecent()
y_pred = np.matmul(gd.X, gd.theta)
print(gd.theta)
plt.figure("Loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.plot([i + 1 for i in range(gd.times)], gd.losslist)
plt.show()


# ## 使用飞桨框架进行房价预测

# In[18]:


## 导库
import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt


# In[19]:


BUF_SIZE=500
BATCH_SIZE=20


# In[20]:


#用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), 
                          buf_size=BUF_SIZE),                    
    batch_size=BATCH_SIZE)   
#用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)  


# In[21]:



#用于打印，查看uci_housing数据
train_data=paddle.dataset.uci_housing.train();
sampledata=next(train_data())
print(sampledata)


# In[23]:


paddle.enable_static()
#定义张量变量x，表示13维的特征值
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
#定义张量y,表示目标值
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
#定义一个简单的线性网络,连接输入和输出的全连接层
#input:输入tensor;
#size:该层输出单元的数目
#act:激活函数
y_predict=fluid.layers.fc(input=x,size=1,act=None)


# In[24]:



cost = fluid.layers.square_error_cost(input=y_predict, label=y) #求一个batch的损失值
avg_cost = fluid.layers.mean(cost)                              #对损失值求平均值


# In[25]:



optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)


# In[26]:



test_program = fluid.default_main_program().clone(for_test=True)


# In[27]:


use_cuda = False                         #use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU           
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)              #创建一个Executor实例exe
exe.run(fluid.default_startup_program()) #Executor的run()方法执行startup_program(),进行参数初始化


# In[28]:


# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])#feed_list:向模型输入的变量表或变量表名


# In[29]:



iter=0;
iters=[]
train_costs=[]

def draw_train_process(iters,train_costs):
    title="training cost"
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs,color='red',label='training cost') 
    plt.grid()
    plt.show()


# In[30]:


EPOCH_NUM=50
model_save_dir = "/home/aistudio/work/fit_a_line.inference.model"

for pass_id in range(EPOCH_NUM):                                  #训练EPOCH_NUM轮
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):              #遍历train_reader迭代器
        train_cost = exe.run(program=fluid.default_main_program(),#运行主程序
                             feed=feeder.feed(data),              #喂入一个batch的训练数据，根据feed_list和data提供的信息，将输入数据转成一种特殊的数据结构
                             fetch_list=[avg_cost])    
        if batch_id % 40 == 0:
            print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))    #打印最后一个batch的损失值
        iter=iter+BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])
       
   
    # 开始测试并输出最后一个batch的损失值
    test_cost = 0
    for batch_id, data in enumerate(test_reader()):               #遍历test_reader迭代器
        test_cost= exe.run(program=test_program, #运行测试cheng
                            feed=feeder.feed(data),               #喂入一个batch的测试数据
                            fetch_list=[avg_cost])                #fetch均方误差
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost[0][0]))     #打印最后一个batch的损失值
    
    #保存模型
    # 如果保存路径不存在就创建
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print ('save models to %s' % (model_save_dir))
#保存训练参数到指定路径中，构建一个专门用预测的program
fluid.io.save_inference_model(model_save_dir,   #保存推理model的路径
                                  ['x'],            #推理（inference）需要 feed 的数据
                                  [y_predict],      #保存推理（inference）结果的 Variables
                                  exe)              #exe 保存 inference model
draw_train_process(iters,train_costs)


# In[31]:


infer_exe = fluid.Executor(place)    #创建推测用的executor
inference_scope = fluid.core.Scope() #Scope指定作用域


# In[32]:


infer_results=[]
groud_truths=[]

#绘制真实值和预测值对比图
def draw_infer_result(groud_truths,infer_results):
    title='Boston'
    plt.title(title, fontsize=24)
    x = np.arange(1,20) 
    y = x
    plt.plot(x, y)
    plt.xlabel('ground truth', fontsize=14)
    plt.ylabel('infer result', fontsize=14)
    plt.scatter(groud_truths, infer_results,color='green',label='training cost') 
    plt.grid()
    plt.show()


# In[33]:


with fluid.scope_guard(inference_scope):#修改全局/默认作用域（scope）, 运行时中的所有变量都将分配给新的scope。
    #从指定目录中加载 推理model(inference model)
    [inference_program,                             #推理的program
     feed_target_names,                             #需要在推理program中提供数据的变量名称
     fetch_targets] = fluid.io.load_inference_model(#fetch_targets: 推断结果
                                    model_save_dir, #model_save_dir:模型训练路径 
                                    infer_exe)      #infer_exe: 预测用executor
    #获取预测数据
    infer_reader = paddle.batch(paddle.dataset.uci_housing.test(),  #获取uci_housing的测试数据
                          batch_size=200)                           #从测试数据中读取一个大小为200的batch数据
    #从test_reader中分割x
    test_data = next(infer_reader())
    test_x = np.array([data[0] for data in test_data]).astype("float32")
    test_y= np.array([data[1] for data in test_data]).astype("float32")
    results = infer_exe.run(inference_program,                              #预测模型
                            feed={feed_target_names[0]: np.array(test_x)},  #喂入要预测的x值
                            fetch_list=fetch_targets)                       #得到推测结果 
                            
    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val))
        infer_results.append(val)
    print("ground truth:")
    for idx, val in enumerate(test_y):
        print("%d: %.2f" % (idx, val))
        groud_truths.append(val)
    draw_infer_result(groud_truths,infer_results)


# In[ ]:





# In[ ]:





# In[ ]:




