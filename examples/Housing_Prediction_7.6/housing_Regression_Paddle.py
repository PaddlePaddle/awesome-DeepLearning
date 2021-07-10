#!/usr/bin/env python
# coding: utf-8

# # 使用飞桨构建波士顿房价预测模型
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/0fb9f19454974a689525cdbecf28fde7d8284b294ffa4a7487b1d3354a88d6f6" width="800" hegiht="" ></center>
# <center><br>图1：使用飞桨框架构建神经网络过程</br></center>
# <br></br>

# In[1]:


#加载飞桨、Numpy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import os
import random


# 代码中参数含义如下：
# 
# * paddle：飞桨的主库，paddle 根目录下保留了常用API的别名，当前包括：paddle.tensor、paddle.framework目录下的所有API。
# 
# * paddle.nn：组网相关的API，例如 Linear 、卷积 Conv2D 、 循环神经网络 LSTM 、损失函数 CrossEntropyLoss 、 激活函数 ReLU 等。
# 
# * Linear：神经网络的全连接层函数，即包含所有输入权重相加的基本神经元结构。在房价预测任务中，使用只有一层的神经网络（全连接层）来实现线性回归模型。
# 
# * paddle.nn.functional：与paddle.nn一样，包含组网相关的API，例如Linear、激活函数ReLu等。两者下的同名模块功能相同，运行性能也基本一致。 但是，paddle.nn下的模块均是类，每个类下可以自带模块参数；paddle.nn.functional下的模块均是函数，需要手动传入模块计算需要的参数。在实际使用中，卷积、全连接层等层本身具有可学习的参数，建议使用paddle.nn模块，而激活函数、池化等操作没有可学习参数，可以考虑直接使用paddle.nn.functional下的函数代替。

# ## 数据处理
# 
# 数据处理的代码不依赖框架实现，与使用Python构建房价预测任务的代码相同。

# In[2]:


def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

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

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0),                                  training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# ## 模型设计
# 
# 模型定义的实质是定义线性回归的网络结构，``forward``函数是框架指定实现前向计算逻辑的函数，程序在调用模型实例时会自动执行forward方法。在``forward``函数中使用的网络层需要在``init``函数中声明。实现过程分如下两步：
# 1. **定义init函数**：在类的初始化函数中声明每一层网络的实现函数。在本次房价预测模型中，定义了：**全连接层** + **节点数为10、激活函数为ReLU的隐层** + **全连接层**。
# 1. **定义forward函数**：构建神经网络结构，实现前向计算过程，并返回预测结果，在本任务中返回的是房价预测结果。
# <br></br>
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/6f70665ec16f45949b60256e30c3f336a8e01ba535084fef98ce7754ae6942ea" width="800" hegiht="" ></center>
# <center><br>图2：房价预测模型网络结构</br></center>
# <br></br>
# 

# In[3]:


#构建网络 隐藏节点数为10 激活函数ReLU

class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(13, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=13, out_features=1)
    
    # 网络的前向计算
    def forward(self, inputs):
        x = self.predict(inputs)
        return x


# ## 训练配置
# 训练配置过程包含四步，如图3所示：
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/b60532f4cd174c1eb621c5b9a006e0b05eaa213f44704d21998b57189c4cc11b" width="700" hegiht="" ></center>
# <center><br>图3：训练配置流程示意图</br></center>
# <br></br>
# 
# 1. 声明定义好的回归模型Regressor实例，并将模型的状态设置为训练。
# 1. 使用load_data函数加载训练数据和测试数据。
# 1. 设置优化算法和学习率，优化算法采用随机梯度下降SGD，学习率设置为0.01。
# 
# 训练配置代码如下所示：

# In[4]:


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 加载数据
training_data, test_data = load_data()
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())


# ## 训练过程
# 训练过程采用二层循环嵌套方式：
# 
# - **内层循环：** 负责整个数据集的一次遍历，采用分批次方式（batch）。假设数据集样本数量为1000，一个批次有10个样本，则遍历一次数据集的批次数量是1000/10=100，即内层循环需要执行100次。
# 
# - **外层循环：** 定义遍历数据集的次数，通过参数EPOCH_NUM设置。
# ------
# **说明**:
# 
# batch的取值会影响模型训练效果。batch过大，会增大内存消耗和计算时间，且训练效果并不会明显提升（因为每次参数只向梯度反方向移动一小步，所以方向没必要特别精确）；batch过小，每个batch的样本数据将没有统计意义，计算的梯度方向可能偏差较大。
# 由于房价预测模型的训练数据集较小，我们将batch为设置10。
# 
# ------
# 
# 每次内层循环都需要执行如下四个步骤，如图4所示，计算过程与使用Python编写模型完全一致。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/6515c82affed4ce4b7128b9f22c78698d8df50b00d21472ea9cca8f974e80b94" width="700" hegiht="" ></center>
# <center><br>图4：内循环计算过程</br></center>
# <br></br>
# 
# 1. 数据准备：将一个批次的数据先转换成np.array格式，再转换成paddle内置tensor格式。
# 1. 前向计算：将一个批次的样本数据灌入网络中，计算输出结果。
# 1. 计算损失函数：以前向计算结果和真实房价作为输入，通过损失函数square_error_cost API计算出损失函数值（Loss）。
# 1. 反向传播：执行梯度反向传播``backward``函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数。

# In[5]:


EPOCH_NUM = 10   # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1]) # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:]) # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        
        # 前向计算
        predicts = model(house_features)
        
        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id%20==0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        
        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()


# ## 保存并测试模型
# 
# ### 保存模型
# 
# 将模型当前的参数数据``model.state_dict()``保存到文件中（通过参数指定保存的文件名 LR_model），以备预测或校验的程序调用，代码如下所示。

# In[6]:


# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), 'LR_model.pdparams')
print("模型保存成功，模型参数保存在LR_model.pdparams中")


# ### 测试模型
# 
# 测试过程主要可分成如下三个步骤：
# 
# 1. 配置模型预测的机器资源。
# 1. 将训练好的模型参数加载到模型实例中。校验和预测状态的模型只需要支持前向计算，模型的实现更加简单，性能更好。
# 1. 将待预测的样本特征输入到模型中，打印输出的预测结果。
# 通过``load_one_example``函数实现从数据集中抽一条样本作为测试样本，具体实现代码如下所示。

# In[7]:


def load_one_example():
    # 从上边已加载的测试集中，随机选择一条作为测试数据
    idx = np.random.randint(0, test_data.shape[0])
    idx = -10
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    # 修改该条数据shape为[1,13]
    one_data =  one_data.reshape([1,-1])

    return one_data, label


# In[8]:


# 参数为保存模型参数的文件地址
model_dict = paddle.load('LR_model.pdparams')
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
one_data, label = load_one_example()
# 将数据转为动态图的variable格式 
one_data = paddle.to_tensor(one_data)
predict = model(one_data)

# 对结果做反归一化处理
predict = predict * (max_values[-1] - min_values[-1]) + avg_values[-1]
# 对label数据做反归一化处理
label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]

print("Inference result is {}, the corresponding label is {}".format(predict.numpy(), label))

