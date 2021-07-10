#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# In[ ]:


import paddle
print(paddle.__version__)


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


import paddle.vision.transforms as T
transform = T.Normalize(mean=[127.5], std=[127.5], data_format='CHW')

# 下载数据集
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
val_dataset =  paddle.vision.datasets.MNIST(mode='test', transform=transform)


# In[ ]:


from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
Resnet18 = paddle.vision.models.ResNet(BasicBlock, depth=18, num_classes=10, with_pool=True)


# In[12]:


# 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
model = paddle.Model(mnist)

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
                paddle.nn.CrossEntropyLoss(),
                paddle.metric.Accuracy())

# 开始模型训练
model.fit(train_dataset,
            epochs=5,
            batch_size=64,
            verbose=1)


# In[13]:


model.evaluate(val_dataset, verbose=0)

