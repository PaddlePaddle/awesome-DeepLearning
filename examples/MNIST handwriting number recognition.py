#!/usr/bin/env python
# coding: utf-8

# **MNIST手写数字识别**
# **一、数据预处理**
# （1）MNIST数据集介绍
# MNIST数据集包含60000个训练集和10000测试数据集。分为图片和标签，图片是28*28的像素矩阵，标签为0~9共10个数字。
# 其中：
# transform函数是定义了一个归一化标准化的标准；
# train_dataset和test_dataset；
# paddle.vision.datasets.MNIST()中的mode='train'和mode='test'分别用于获取mnist训练集和测试集；
# transform=transform参数则为归一化标准

# In[11]:


import paddle


# In[12]:


#导入数据集Compose的作用是将用于数据集预处理的接口以列表的方式进行组合。
#导入数据集Normalize的作用是图像归一化处理，支持两种方式： 1. 用统一的均值和标准差值对图像的每个通道进行归一化处理； 2. 对每个通道指定不同的均值和标准差值进行归一化处理。
from paddle.vision.transforms import Compose, Normalize, Resize, RandomRotation, RandomCrop
img_size = 32
#对训练集做数据增强
transform1 = Compose([Resize((img_size+2, img_size+2)), RandomCrop(img_size), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
transform2 = Compose([Resize((img_size, img_size)), Normalize(mean=[127.5],std=[127.5],data_format='CHW')])
# 使用transform对数据集做归一化
print('下载并加载训练数据')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform1)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform2)
print('加载完成')


# **二、搭建网络模型**
# 本次所使用的网络共有7层：5个卷积层、2个全连接层；
#    其中第一个卷积层的输入通道数为数据集图片的实际通道数。MNIST数据集为灰度图像，通道数为1；
#    第1个卷积层输出与第3个卷积层输出做残差作为第4个卷积层的输入，第4个卷积层的输入与第5个卷积层的输出做残差作为第1个全连接层的输入；
# 代码如下：

# In[13]:


import paddle
import paddle.nn.functional as F
# 定义多层卷积神经网络
#动态图定义多层卷积神经网络
class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = paddle.nn.Conv2D(6, 16, 3, padding=1)
        self.conv3 = paddle.nn.Conv2D(16, 32, 3, padding=1)
        self.conv4 = paddle.nn.Conv2D(6, 32, 1)
        self.conv5 = paddle.nn.Conv2D(32, 64, 3, padding=1)
        self.conv6 = paddle.nn.Conv2D(64, 128, 3, padding=1)
        self.conv7 = paddle.nn.Conv2D(32, 128, 1)
        self.maxpool1 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.maxpool2 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool3 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool4 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool5 = paddle.nn.MaxPool2D(2, 2)
        self.maxpool6 = paddle.nn.MaxPool2D(2, 2)
        self.flatten=paddle.nn.Flatten()
        self.linear1=paddle.nn.Linear(128, 128)
        self.linear2=paddle.nn.Linear(128, 10)
        self.dropout=paddle.nn.Dropout(0.2)
        self.avgpool=paddle.nn.AdaptiveAvgPool2D(output_size=1)
        
    def forward(self, x):
        y = self.conv1(x)#(bs 6, 32, 32)
        y = F.relu(y)
        y = self.maxpool1(y)#(bs, 6, 16, 16)
        z = y
        y = self.conv2(y)#(bs, 16, 16, 16)
        y = F.relu(y)
        y = self.maxpool2(y)#(bs, 16, 8, 8)
        y = self.conv3(y)#(bs, 32, 8, 8)
        z = self.maxpool4(self.conv4(z))
        y = y+z
        y = F.relu(y)
        z = y
        y = self.conv5(y)#(bs, 64, 8, 8)
        y = F.relu(y)
        y = self.maxpool5(y)#(bs, 64, 4, 4)
        y = self.conv6(y)#(bs, 128, 4, 4)
        z = self.maxpool6(self.conv7(z))
        y = y + z
        y = F.relu(y)
        y = self.avgpool(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return y


# In[14]:


#定义卷积网络的代码
net_cls = MyNet()
paddle.summary(net_cls, (-1, 1, img_size, img_size))


# **三、训练模型**

# In[15]:


from paddle.metric import Accuracy
save_dir = "output/model/v5_7"
patience = 5
epoch = 50
lr = 0.01
weight_decay = 5e-4
batch_size = 64
momentum = 0.9
# 用Model封装模型
model = paddle.Model(net_cls)

# 定义损失函数
#optim = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(), weight_decay=weight_decay)
#lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=lr, T_max=10000, eta_min=1e-5)
optim = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters(), momentum=momentum)

visual_dl = paddle.callbacks.VisualDL(log_dir=save_dir)
early_stop = paddle.callbacks.EarlyStopping(monitor='acc', mode='max', patience=patience, 
                                            verbose=1, min_delta=0, baseline=None,
                                            save_best_model=True)
# 配置模型
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())

# 训练保存并验证模型
model.fit(train_dataset,test_dataset,epochs=epoch,batch_size=batch_size,
        save_dir=save_dir,verbose=1, callbacks=[visual_dl, early_stop])


# 四、测试模型

# In[18]:


best_model_path = "output/model/v5_7/final.pdparams"
net_cls = MyNet()
model = paddle.Model(net_cls)
model.load(best_model_path)
model.prepare(optim,paddle.nn.CrossEntropyLoss(),Accuracy())


# In[19]:


#用最好的模型在测试集10000张图片上验证
results = model.evaluate(test_dataset, batch_size=batch_size, verbose=1)
print(results)


# In[21]:


from matplotlib import pyplot as plt 
########测试
#获取测试集的第一个图片
test_data0, test_label_0 = test_dataset[0][0],test_dataset[0][1]
test_data0 = test_data0.reshape([img_size,img_size])
plt.figure(figsize=(2,2))
#展示测试集中的第一个图片
print(plt.imshow(test_data0, cmap=plt.cm.binary))
print('test_data0 的标签为: ' + str(test_label_0))
#模型预测
result = model.predict(test_dataset, batch_size=1)
#打印模型预测的结果
print('test_data0 预测的数值为：%d' % np.argsort(result[0][0])[0][-1])

