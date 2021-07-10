#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# ### **1. ResNet**
# 
# &emsp;&emsp;LeNet 和 AlexNet的提出开启了卷积神经网络应用的先河，随后的GoogleNet、VGG等网络使用了更小的卷积核并加大了深度，证明了卷积神经网络在处理图像问题方面具有更加好的性能；但是随着层数的不断加深，卷积神经网络也暴露出来许多问题：
# 
# &emsp;&emsp;1.理论上讲，层数越多、模型越复杂，其性能就应该越好；但是实验证明随着层数的不断加深，性能反而有所下降。
# 
# &emsp;&emsp;2.深度卷积网络往往存在着梯度消失/梯度爆炸的问题；由于梯度反向传播过程中，如果梯度都大于1，则每一层大于1的梯度会不断相乘，使梯度呈指数型增长；同理如果梯度都小于1，梯度则会逐渐趋于零；使得深度卷积网络难以训练。
# 
# &emsp;&emsp;3.训练深层网络时会出现退化：随着网络深度的增加，准确率达到饱和，然后迅速退化。
# 
# &emsp;&emsp;而ResNet提出的残差结构，则一定程度上缓解了模型退化和梯度消失问题：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/6f8bf8f0bc2146a59e903d757b4614a0513b4a735ec94bc9ab602cf27c42996b)
# 
# &emsp;&emsp;作者提出，在一个结构单元中，如果我们想要学习的映射本来是y=H(x)，那么跟学习y=F(x)+x这个映射是等效的；这样就将本来回归的目标函数H(x)转化为F(x)+x，即F(x) = H(x) - x，称之为残差。
# 
# &emsp;&emsp;于是，ResNet相当于将学习目标改变了，不再是学习一个完整的输出，而是目标值H(x)和x的差值，即去掉映射前后相同的主体部分，从而突出微小的变化，也能够将不同的特征层融合。而且y=F(x)+x在反向传播求导时，x项的导数恒为1这样也解决了梯度消失问题。
# 
# 

# ### **2.DenseNet**
# 
# &emsp;&emsp;DenseNet 的主要思想是将每一层都与后面的层都紧密（Dense）连接起来，将特征图重复利用，网络更窄，参数更少，对特征层能够更有效地利用和传递，并减轻了梯度消失的问题。
# 
# &emsp;&emsp;网络结构如图：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/94a4783006be4b018f37c10b09396e2ce8a52acc0e294e9d9c65eda1e5daa9ef)
# 
# &emsp;&emsp;其基本的结构单元为：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/1a769f62b5e143508ddeb06ddcd9c2ba697b46760c2c4b038541a6502d0c8386)
# 

# ### **代码实现**

# In[2]:


#导入需要的包
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import matplotlib.pyplot as plt
import os


# In[3]:


BUF_SIZE=128 # 每次缓存队列中保存数据的个数
BATCH_SIZE=32 # 批次大小

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE) # paddle 给的数据迭代器

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.test(),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)    

train_data=paddle.dataset.mnist.train();  


# In[6]:


# 定义DenseNet

class DenseNet(): 
    def __init__(self, layers, dropout_prob):
        self.layers = layers
        self.dropout_prob = dropout_prob

    def bottleneck_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num * 4, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        bn = fluid.layers.batch_norm(input=dropout, act='relu', name=name + '_bn2')
        conv2 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=3, padding=1, name=name + '_conv2')
        dropout = fluid.layers.dropout(x=conv2, dropout_prob=self.dropout_prob)

        return dropout

    def dense_block(self, input, block_num, fliter_num, name):
        layers = []
        layers.append(input)#拼接到列表

        x = self.bottleneck_layer(input, fliter_num, name=name + '_bottle_' + str(0))
        layers.append(x)
        for i in range(block_num - 1):
            x = paddle.fluid.layers.concat(layers, axis=1)
            x = self.bottleneck_layer(x, fliter_num, name=name + '_bottle_' + str(i + 1))
            layers.append(x)

        return paddle.fluid.layers.concat(layers, axis=1)

    def transition_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=1, name=name + '_conv1') 
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        return fluid.layers.pool2d(input=dropout, pool_size=2, pool_type='avg', pool_stride=2) 
    def net(self, input, class_dim=1000): 

        layer_count_dict = {
            9: (32, [3, 3, 6])
        }
        layer_conf = layer_count_dict[self.layers]

        conv = fluid.layers.conv2d(input=input, num_filters=layer_conf[0] * 2, 
            filter_size=3, name='densenet_conv0')
        conv = fluid.layers.pool2d(input=conv, pool_size=2, pool_padding=1, pool_type='max', pool_stride=2)
        for i in range(len(layer_conf[1]) - 1):
            conv = self.dense_block(conv, layer_conf[1][i], layer_conf[0], 'dense_' + str(i))
            conv = self.transition_layer(conv, layer_conf[0], name='trans_' + str(i))

        conv = self.dense_block(conv, layer_conf[1][-1], layer_conf[0], 'dense_' + str(len(layer_conf[1])))
        conv = fluid.layers.pool2d(input=conv, global_pooling=True, pool_type='avg')
        out = fluid.layers.fc(conv, class_dim, act='softmax')

        return out


# In[7]:


# 定义输入输出层
paddle.enable_static()
image = fluid.layers.data(name='image', shape=[1, 28, 28], dtype='float32')#单通道，28*28像素值
label = fluid.layers.data(name='label', shape=[1], dtype='int64') # 图片标签
# 获取分类器
model = DenseNet(9, 0.5)
out = model.net(input=image, class_dim=10)
# 获取损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=out, label=label)  #使用交叉熵损失函数,描述真实样本标签和预测概率之间的差值
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=out, label=label) # 定义准确率


# In[8]:


# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)   #使用Adam算法进行优化
opts = optimizer.minimize(avg_cost)
# 定义一个使用CPU的解析器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(place=place, feed_list=[image, label])


# In[9]:


all_train_iter=0
all_train_iters=[]
all_train_costs=[]
all_train_accs=[]

#绘制训练时loss图像
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()


# In[ ]:


EPOCH_NUM=1  # 调参 训练轮数
for pass_id in range(EPOCH_NUM):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):                         #遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
                                        feed=feeder.feed(data),              #给模型喂入数据
                                        fetch_list=[avg_cost, acc])          #fetch 误差、准确率                                          
        all_train_iter=all_train_iter+1
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])        
        # 每100个batch打印一次信息  误差、准确率
        if batch_id % 200 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

    # 进行测试
    test_accs = []
    test_costs = []
    #每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):                         #遍历test_reader
        test_cost, test_acc = exe.run(program=fluid.default_main_program(), #执行训练程序
                                      feed=feeder.feed(data),               #喂入数据
                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
        test_accs.append(test_acc[0])                                       #每个batch的准确率
        test_costs.append(test_cost[0])                                     #每个batch的误差                              
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))                         #每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))                            #每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))    
draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")

