# 百度第一次作业

### 1.深度学习发展历史

* **第一代神经网络（1958-1969）：**MCP神经元数学模型（1943）→单层感知机（1957）→单层感知机不能实现XOR问题（1969）

* **第二代神经网络（1986-1998）：**BP算法（1986）→卷积神经网络-LeNet（1998）

* **统计学习方法（1986-2006）：**决策树方法（1986）→线性SVM（1995）→AdaBoost（1997）→KernelSVM（2000）→随机森林（2001）→图模型（2001）

* **第三代神经网络-DL（2006-至今）：**快速发展期（2006-2012）→爆发期（2012-至今）

### 2.人工智能、机器学习、深度学习有什么区别和联系

* 深度学习∈机器学习∈人工智能

### 3.神经元、单层感知机、多层感知机

* **神经元：**仿照生物神经元的一个数学模型，如图所示

![See the source image](https://pic3.zhimg.com/v2-b6420b9bcddd4842f04394e8cd0d7b36_r.jpg)

* **单层感知机：**单层感知机是二分类的线性分类模型，输入是被感知数据集的特征向量，输出时数据集的类别{+1,-1}。

![See the source image](https://n.sinaimg.cn/spider202014/26/w1080h546/20200104/917c-imrkkfx5939001.png)

* **多层感知机：**即人工神经网络，除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构。

![See the source image](https://th.bing.com/th/id/R.c2ce0e7110c4bbe44c095f78f8246e2f?rik=1c9v6RAJvGUxDA&riu=http%3a%2f%2fupload-images.jianshu.io%2fupload_images%2f749674-1f47a199a6ce5008.png&ehk=pCOcoxMTgQHivRXnJG9A20VhNXbpz97FVqsyqJndtG4%3d&risl=&pid=ImgRaw)

### 4.什么是前向传播（包含图文示例）

网络如何根据输入X得到输出Y的过程就是前向传播。

![See the source image](https://th.bing.com/th/id/R.bfc419b3cb4508be3ef404601008cf10?rik=l%2f7l3ZVcLqbQTQ&riu=http%3a%2f%2fimg.blog.csdn.net%2f20160515204217870&ehk=EeeH4pRppj284HGJdPUKq7aN4%2f7MGTuB3WAR5RbgFks%3d&risl=&pid=ImgRaw)

### 5.什么是反向传播（包含图文示例）

* 通过前向传播得到由任意一组随机参数W和b计算出的网络预测结果后，我们可以利用损失函数相对于每个参数的梯度来对他们进行修正。

![See the source image](https://pic1.zhimg.com/v2-9a4aff21fc12d343cc3a4f1c663e8c91_1200x500.jpg)

* 1、计算总误差→隐含层---->2、输出层的权值更新→3、隐含层---->隐含层的权值更新

