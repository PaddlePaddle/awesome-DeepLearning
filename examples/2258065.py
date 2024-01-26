#!/usr/bin/env python
# coding: utf-8

# # DenseNet结构及原理概述
# ## 1.网络特点
# 深度学习网络中，随着网络深度的加深，梯度消失的问题会越来越明显。ResNet，Highway Networks，Stochastic depth，FractalNets等网络都在不同方面针对这个问题提出解决方案，但核心方法都是**建立浅层与深层之间的连接**。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/0511a5d8389644e3bc31cccab413e01bada1b1abf13e41719e8e2e7830d51a10)
# 图1. DenseNet连接机制
# 
# DenseNet继续延申了这一思想，将当前层与之前所有层连接起来，上图即为一个Dense Block。
# DenseNet的一个优点就是网络更窄，参数更少，并且特征和梯度的传递更有效，网络也就更容易训练。
# ## 2.结构对比
# ![](https://ai-studio-static-online.cdn.bcebos.com/38e80ce2f5154806b30de6f0777f951f4ab432550535445ca0587cd46a54ddcf)
# ## 3.传递公式
# 先看ResNet的：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/b324ab87677740af8939bcac93f62104d727dbd5d84243c1ae58fc5189bff1f1)
# 
# 将上一层的输入和输出相加得到这一层的输入。
# 而DenseNet:
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/343f7c85401545698e88d241d4583093fa30c3bde32f4f149f532f70ea934c15)
# 
# 这一层的输入是之前的所有层，从式子中就能清晰地看出DenseNet的运作方式以及和ResNet的差别。
# ## 4.网络整体结构图
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/6537a31b773943f9af75c534a9a779ccadef505cb29f4a67a9937894946059a7)
# 
# 图中包含3个Dense Block,可以看到每个Dense Block中的所有层都与其之前的每一层相连。每两个Dense Block之间还有一个$1 \times 1$卷积层和一个$2 \times 2$池化层,这是为了减少输入的feature map，降维减少计算量，融合各通道特征。
# 每张图片先经过卷积输入，然后经过几个Dense Block，最后再经过一次卷积输入到全连接层中，输出（分类）结果。
# ## 5.几种常用的结构
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/2918cb1651f14367a68f52aa165b6e7101bdb91c521a42678cfba367f1c24620)
# 
# 图中的k=32和k=48表示每个Dense Block中每层输出的feature map个数，作者的实验表明32或48这种较小的k会有更好的效果。
# ## 6.网络效果
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/0c287b101fc543f6a89599b7ce32e0d40637f573e07c46bfa45b84a66139942c)
# 
# Table2是在三个数据集（C10，C100，SVHN）上和其他算法的对比结果。ResNet[11]就是kaiming He的论文，对比结果一目了然。DenseNet-BC的网络参数和相同深度的DenseNet相比确实减少了很多！参数减少除了可以节省内存，还能减少过拟合。这里对于SVHN数据集，DenseNet-BC的结果并没有DenseNet(k=24)的效果好，作者认为原因主要是SVHN这个数据集相对简单，更深的模型容易过拟合。在表格的倒数第二个区域的三个不同深度L和k的DenseNet的对比可以看出随着L和k的增加，模型的效果是更好的。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/7c4c1f964aca4b0795e4b9e7eaa00c3f2f41c12eb7c04f8181a67000eaee2473)
# 
# 左图是参数复杂度和错误率的对比，可以在相同错误率下看参数复杂度，也可以在相同参数复杂度下看错误率，提升还是很明显的。右边是flops（可以理解为计算复杂度）和错误率的对比，同样有效果。
# DenseNet核心思想在于建立了不同层之间的连接关系，充分利用了feature，进一步减轻了梯度消失问题，加深网络不是问题，而且训练效果非常好。
# 
# 参考文献：[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

# # 代码实现
# ## 1. 实验设计逻辑
# ### 解释任务，说明实验设计逻辑
# 任务要求是在眼疾识别数据集上训练DenseNet网络，实现分类的效果。
# 根据要求构建Densenet网络并使用眼疾识别数据集中的训练集进行训练，然后再用测试集测试训练效果。

# ## 2. 数据处理
# ### 解释数据集，处理数据为模型输入格式
# iChallenge-PM是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。  
# training.zip：包含训练中的图片和标签。  
# validation.zip：包含验证集的图片。  
# valid_gt.zip：包含验证集的标签。  
# iChallenge-PM中既有病理性近视患者的眼底图片，也有非病理性近视患者的图片，命名规则如下：  
# 病理性近视（PM）：文件名以P开头。  
# 非病理性近视（non-PM）： 高度近似（high myopia）：文件名以H开头。  
# 正常眼睛（normal）：文件名以N开头。  
# 处理数据集：首先解压数据集，根据数据集介绍，训练集images通过文件名获取相应的labels,测试集通过读取PM_Label_and_Fovea_Location.xlsx文件获取文件名和label信息，然后分别生成traindata.txt和valdata.txt文件。使用时直接读入文件即可获取图片与标签对应关系。

# In[1]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[1]:


#解压数据集 已经解压到WORK就不用再解压了
get_ipython().system('unzip data/data23828/training.zip -d work/')
get_ipython().system('unzip data/data23828/valid_gt.zip -d work/')
get_ipython().system('unzip data/data23828/validation.zip -d work/')


# In[48]:


#解压数据集 已经解压到WORK就不用再解压了
get_ipython().system('unzip work/PALM-Training400/PALM-Training401.zip -d work/')


# In[64]:


import os
import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()。用在Jupyter notebook中具体作用是调用matplotlib.pyplot的绘图函数plot()进行绘图的时候，或者生成一个figure画布的时候，可以直接在你的python console里面生成图像。
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image

DATADIR = 'work/PALM-Training400/PALM-Training400'
# 文件名以N开头的是正常眼底图片，以P开头的是病变眼底图片
file1 = 'N0001.jpg'
file2 = 'P0001.jpg'

# 读取图片
img1 = Image.open(os.path.join(DATADIR, file1))
img1 = np.array(img1)
img2 = Image.open(os.path.join(DATADIR, file2))
img2 = np.array(img2)

# 画出读取的图片
plt.figure(figsize=(16, 8)) # 设置画布
# 位置是由三个整型数值构成，第一个代表行数，第二个代表列数，第三个代表索引位置。举个列子：plt.subplot(2, 3, 5) 和 plt.subplot(235) 是一样一样的。需要注意的是所有的数字不能超过10。
f = plt.subplot(121) 
f.set_title('normal', fontsize=20)
plt.imshow(img1)
f = plt.subplot(122)
f.set_title('PM', fontsize=20)
plt.imshow(img2)
plt.show()


# In[7]:


#生成label文本
import os
import numpy as np
path = ''
trainpath = 'work/PALM-Training400'
imgdirs = os.listdir(os.path.join(path, trainpath))
traindata = open('traindata.txt', 'w', encoding = 'utf-8')
for file in imgdirs:
    if '.jpg' in file:
        traindata.write(trainpath + '/' + file + ' ')
        if file[0] is 'H' or file[0] is 'N':
            traindata.write('0')
        else:
            traindata.write('1')
        traindata.write('\n')
traindata.flush()
traindata.close()


# In[6]:


#生成验证集label文本，通过读取xlsx文件
import os
import numpy as np
from openpyxl import load_workbook
path = ''
valpath = 'work/PALM-Validation400'
vallabelpath = 'work/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx'
oldfile = load_workbook(os.path.join(path, vallabelpath))
newfile = open('valdata.txt', 'w', encoding = 'utf-8')
sheet = oldfile.worksheets[0]
rows = sheet.rows
for row in sheet[2:401]:
    newfile.write(valpath + '/' + row[1].value + ' ' + str(row[2].value) + '\n')
newfile.flush()
newfile.close()


# In[65]:


get_ipython().run_line_magic('cd', 'work/')
#%cd ..


# ## 3. 模型设计
# ### 根据任务设计模型，需要给出模型设计图
# 这里采用DenseNet-121结构
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/84fb8c9feb244de0b792eb1e41fc03b68a8128cb78ff4eedaf1289fe9e17bb01)
# ![](https://ai-studio-static-online.cdn.bcebos.com/9c956633a9d0409fb682904c87e68d4b50243130cf0e4845b369726283df2ebb)
# ![](https://ai-studio-static-online.cdn.bcebos.com/1a3ae1465e26463abf4ea35b2a14e299df2ddcec6a2a440c8b075251e7cf00cc)
# 
# 模型设计在net.py中

# In[9]:


get_ipython().system('python net.py')


# ## 4. 定义数据读取器
# ### 自定义用户图片读取器，先初始化图片种类，数量，定义图片增强和强制缩放函数
# 在reader.py中

# In[8]:


get_ipython().system('python reader.py')


# ## 5. 训练配置
# ### 定义模型训练的超参数，模型实例化，指定训练的 cpu 或 gpu 资 源，定义优化器等等
# 这里将训练的超参数写入了config.py代码中，里面包含了各类超参数以及初始化训练参数的函数

# In[60]:


get_ipython().system('python config.py')


# ## 6. 模型训练与评估
# ### 训练模型，在训练过程中，根据开发集适时打印结果
# 在train.py中自定义模型训练，在eval.py中定义模型评估
# 
# 模型文件保存在./model中 （work目录下）
# 
# 日志文件保存在./logs/train.log中(work目录下)

# In[50]:


get_ipython().system('python train.py')


# In[58]:


get_ipython().system('python eval.py')


# ## 7. 模型推理
# ### 设计一个接口函数，通过这个接口函数能够方便地对任意一个样本进行实时预测
# infer.py中 imgpath为要预测的文件路径，直接调用infer_img(imgpath)即可实时预测

# In[61]:


get_ipython().system('python infer.py')

