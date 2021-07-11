#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # 深度学习历史
# 
# 
# 1943年
# 由神经科学家麦卡洛克(W.S.McCilloch) 和数学家皮兹（W.Pitts）在《数学生物物理学公告》上发表论文《神经活动中内在思想的逻辑演算》（A Logical Calculus of the Ideas Immanent in Nervous Activity）。建立了神经网络和数学模型，称为MCP模型。所谓MCP模型，其实是按照生物神经元的结构和工作原理构造出来的一个抽象和简化了的模型，也就诞生了所谓的“模拟大脑”，人工神经网络的大门由此开启。
# MCP当时是希望能够用计算机来模拟人的神经元反应的过程，该模型将神经元简化为了三个过程：输入信号线性加权，求和，非线性激活（阈值法）。如下图所示
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/9ba013f7e81a465583425fa73e51fb16ea662a4c516a4864b0eebacdd0f16ac8" width="400" hegiht="" ></center>
# 1958年计算机科学家罗森布拉特（ Rosenblatt）提出了两层神经元组成的神经网络，称之为“感知器”(Perceptrons)。第一次将MCP用于机器学习（machine learning）分类(classification)。“感知器”算法算法使用MCP模型对输入的多维数据进行二分类，且能够使用梯度下降法从训练样本中自动学习更新权值。1962年,该方法被证明为能够收敛，理论与实践效果引起第一次神经网络的浪潮。
# 
# 1969年纵观科学发展史，无疑都是充满曲折的，深度学习也毫不例外。 1969年，美国数学家及人工智能先驱 Marvin Minsky 在其著作中证明了感知器本质上是一种线性模型（linear model），只能处理线性分类问题，就连最简单的XOR（亦或）问题都无法正确分类。这等于直接宣判了感知器的死刑，神经网络的研究也陷入了将近20年的停滞。
# 
# 1986年由神经网络之父 Geoffrey Hinton 在1986年发明了适用于多层感知器（MLP）的BP（Backpropagation）算法，并采用Sigmoid进行非线性映射，有效解决了非线性分类和学习的问题。该方法引起了神经网络的第二次热潮。
# 
# 注：Sigmoid 函数是一个在生物学中常见的S型的函数，也称为S型生长曲线。在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的阈值函数，将变量映射到0,1之间。
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/395993f005a04d62b04ba6a930482397e038f96c264b4e45afebb86dac7538bd" width="400" hegiht="" ></center>
# 
# 90年代时期
# 1991年BP算法被指出存在梯度消失问题，也就是说在误差梯度后项传递的过程中，后层梯度以乘性方式叠加到前层，由于Sigmoid函数的饱和特性，后层梯度本来就小，误差梯度传到前层时几乎为0，因此无法对前层进行有效的学习，该问题直接阻碍了深度学习的进一步发展。
# 
# 此外90年代中期，支持向量机算法诞生（SVM算法）等各种浅层机器学习模型被提出，SVM也是一种有监督的学习模型，应用于模式识别，分类以及回归分析等。支持向量机以统计学为基础，和神经网络有明显的差异，支持向量机等算法的提出再次阻碍了深度学习的发展。
# 
# 发展期 2006年 - 2012年
# 2006年，加拿大多伦多大学教授、机器学习领域泰斗、神经网络之父—— Geoffrey Hinton 和他的学生 Ruslan Salakhutdinov 在顶尖学术刊物《科学》上发表了一篇文章，该文章提出了深层网络训练中梯度消失问题的解决方案：无监督预训练对权值进行初始化+有监督训练微调。斯坦福大学、纽约大学、加拿大蒙特利尔大学等成为研究深度学习的重镇，至此开启了深度学习在学术界和工业界的浪潮。
# 
# 2011年，ReLU激活函数被提出，该激活函数能够有效的抑制梯度消失问题。2011年以来，微软首次将DL应用在语音识别上，取得了重大突破。微软研究院和Google的语音识别研究人员先后采用DNN技术降低语音识别错误率20％~30％，是语音识别领域十多年来最大的突破性进展。2012年，DNN技术在图像识别领域取得惊人的效果，在ImageNet评测上将错误率从26％降低到15％。在这一年，DNN还被应用于制药公司的DrugeActivity预测问题，并获得世界最好成绩。
# 
# 爆发期 2012 - 2017
# 2012年，Hinton课题组为了证明深度学习的潜力，首次参加ImageNet图像识别比赛，其通过构建的CNN网络AlexNet一举夺得冠军，且碾压第二名（SVM方法）的分类性能。也正是由于该比赛，CNN吸引到了众多研究者的注意。
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/d05fc739a8ee48f899cc3d665490dae4336447e1a3f04ac68aafe950b3dadaf9" width="400" hegiht="" ></center>
# 
# AlexNet的创新点在于:
# 
# (1)首次采用ReLU激活函数，极大增大收敛速度且从根本上解决了梯度消失问题。
# 
# (2)由于ReLU方法可以很好抑制梯度消失问题，AlexNet抛弃了“预训练+微调”的方法，完全采用有监督训练。也正因为如此，DL的主流学习方法也因此变为了纯粹的有监督学习。
# 
# (3)扩展了LeNet5结构，添加Dropout层减小过拟合，LRN层增强泛化能力/减小过拟合。
# 
# (4)第一次使用GPU加速模型计算。
# 
# 2013、2014、2015、2016年，通过ImageNet图像识别比赛，DL的网络结构，训练方法，GPU硬件的不断进步，促使其在其他领域也在不断的征服战场。

# # 人工智能、机器学习、深度学习有什么区别和联系？
# 
# 
# 近些年人工智能、机器学习和深度学习的概念十分火热，但很多从业者却很难说清它们之间的关系，外行人更是雾里看花。在研究深度学习之前，我们先从三个概念的正本清源开始。
# 
# 概括来说，人工智能、机器学习和深度学习覆盖的技术范畴是逐层递减的。人工智能是最宽泛的概念。机器学习是当前比较有效的一种实现人工智能的方式。深度学习是机器学习算法中最热门的一个分支，近些年取得了显著的进展，并替代了大多数传统机器学习算法。三者的关系如 **图1** 所示，即：人工智能 > 机器学习 > 深度学习。
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/5521d1d951c440eb8511f03a0b9028bd63357aec52e94189b5ab3f55d63369d7" width="300" hegiht="" ></center>
# <center><br>图1：人工智能、机器学习和深度学习三者关系示意</br></center>
# <br></br>
# 
# 
# 如字面含义，人工智能是研发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门新的技术科学。由于这个定义只阐述了目标，而没有限定方法，因此实现人工智能存在的诸多方法和分支，导致其变成一个“大杂烩”式的学科。
# 

# # 神经元、单层感知机、多层感知机
# 
# 神经元：这里的神经元指的是人工神经网络中的神经元。神经元是一种处理单元，是对人脑组织的神经元的某种抽象、简化和模拟。是人工神经网络的关键部分。通过神经元，人工神经网络可以以数学模型模拟人脑神经元活动，继而进行高效的计算以及其他处理。
# 
# 单层感知机：将被感知数据集划分为两类的分离超平面，并计算出该超平面。是二分类的线性分类模型，输入是被感知数据集的特征向量，输出时数据集的类别{+1,-1}。感知器的模型可以简单表示为：
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/9527c77e9520474490aff1e343b6509a1e1b4d01ca264c619a124ae11d7228bf" width="400" hegiht="" ></center>
# 
# 多层感知机：多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。以单隐藏层为例并沿用本节之前定义的符号，多层感知机按以下方式计算输出：
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/dafb542b821848739c6390dc20152547f4b2a0075eb740ae989345a4eced0f84" width="400" hegiht="" ></center>
# 其中ϕ \phiϕ表示激活函数。

# # 前向传播
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/41d1f818fb1b4ec5878f778377642b4744056156e0e54d4b8f6e566b09d6f383" width="400" hegiht="" ></center>
# 
# 对于第2层第1个节点的输出有：
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/f4841a6920fe4deebed521d042a98fcd131e04a955ac471ebef59e2e07d0a52a" width="400" hegiht="" ></center>
# 
# 对于第3层第1个节点的输出有：
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/090195555cd6469185e884b1aa486511c5fc176f847a4f32aae80cbaee6494b1" width="400" hegiht="" ></center>
# 
# 一般化的，假设l-1层有m个神经元，对于有
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/1a30651715dd4f66848ba3382ead67703b3c5e55db814a35996df4ebba3a4691" width="400" hegiht="" ></center>
# 
# 
# 也就是第l层第j个神经元的输入为与它相连的上一层每个神经元的输出加权求和后加上该神经元对应的偏置，该神经元所做的工作只是把这个结果做一个非线性激活。
# 

# # 反向传播
# <center><img src="" width="400" hegiht="" ></center>
# 
# 当通过前向传播得到由任意一组随机参数W和b计算出的网络预测结果后，我们可以利用损失函数相对于每个参数的梯度来对他们进行修正。事实上神经网络的训练就是这样一个不停的前向-反向传播的过程，直到网络的预测能力达到我们的预期。
# 
# 
# <center><img src="https://ai-studio-static-online.cdn.bcebos.com/b1ba3e0e897f4531b8d43b4ef57bc90af99b002b9d4142729c56017e0a01a37b" width="400" hegiht="" ></center>
# 

# In[ ]:





# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[12]:





# In[28]:


def deal_data():
    # 读取文件数据，此时数据形状是(870,2)
    
    df =  pd.read_csv("./data/房价预测/data/data.txt",sep=',')  # 用pandas读取数据，分隔符为逗号
    housingdata = np.array(df)  # 将DF数据类型转化为numpy数据类型

    # 规范数据格式。
    housingdata = np.array(housingdata).reshape((-1, 2))  # 此时数据形状为(870,2)

    # 对数据的第一个属性进行归一化操作，有助于提高模型精准度，这里使用max-min归一化方式。公式为(x-min)/(max-min)
    for i in range(1):
        Max = np.max(housingdata[:, i])
        Min = np.min(housingdata[:, i])
        housingdata[:, i] = (housingdata[:, i] - Min) / (Max - Min)

    # 依据2-8原则，80%的数据作为训练数据，20%数据作为测试数据；
    Splitdata = round(len(housingdata) * 0.8)
    Train = housingdata[:Splitdata]  # 训练数据集
    Test = housingdata[Splitdata:]  # 测试数据集
    return Train, Test


# In[29]:


class Model_Config(object):
    def __init__(self, firstnetnum, secondnetnum):
        np.random.seed(1)
        self.w0 = np.random.randn(firstnetnum * secondnetnum, 1).reshape(firstnetnum, secondnetnum)
        self.w1 = np.random.randn(secondnetnum, 1)
        self.b0 = np.random.randn(firstnetnum, 1).reshape(1, firstnetnum)
        self.b1 = np.random.randn(1, 1)

    # 计算预测值，前向传播过程
    def forward(self, x):
        hidden1 = np.dot(x, self.w0) + self.b0  # 由输入层到隐藏层
        y = np.dot(hidden1, self.w1) + self.b1  # 由隐藏层到输出层
        return hidden1, y

    # 设置损失函数,这里使用差平方损失函数计算方式
    def loss(self, z, y):
        error = z - y
        cost = error * error
        avg_cost = np.mean(cost)
        return avg_cost

    # 计算梯度
    def back(self, x, y):
        hidden1, z = self.forward(x)
        # hidden层的梯度
        gradient_w1 = (z - y) * hidden1
        gradient_w1 = np.mean(gradient_w1, axis=0)  # 这里注意，axis=0必须写上，否则默认将这个数组变成一维的求平均
        gradient_w1 = gradient_w1[:, np.newaxis]  #
        gradient_b1 = (z - y)
        gradient_b1 = np.mean(gradient_b1)
        gradient_w0 = np.zeros(shape=(1, 1))
        for i in range(len(x)):
            data = x[i, :]
            data = data[:, np.newaxis]
            # print("data.shape",data.shape)
            w1 = self.w1.reshape(1, 1)
            # print("self.w1.shape",w1.shape)
            gradient_w01 = (z - y)[i] * np.dot(data, w1)
            # print("gradient_w01.shape:",gradient_w01.shape)
            gradient_w0 += gradient_w01
        gradient_w0 = gradient_w0 / len(x)
        w2 = self.w1.reshape(1, 1)
        gradient_b0 = np.mean((z - y) * w2, axis=0)

        return gradient_w1, gradient_b1, gradient_w0, gradient_b0

    # 使用梯度更新权值参数w1，b1,w0,b0
    def update(self, gradient_w1, gradient_b1, gradient_w0, gradient_b0, learning_rate):
        self.w1 = self.w1 - learning_rate * gradient_w1
        self.b1 = self.b1 - learning_rate * gradient_b1
        self.w0 = self.w0 - learning_rate * gradient_w0
        self.b0 = self.b0 - learning_rate * gradient_b0

    # 开始训练
    def train(self, epoch_num, x, y, learning_rate):
        # 循环迭代
        losses = []
        for i in range(epoch_num):
            _, z = self.forward(x)  # 前向传播
            avg_loss = self.loss(z, y)  # 损失函数
            gradient_w1, gradient_b1, gradient_w0, gradient_b0 = self.back(x, y)  # 计算梯度
            self.update(gradient_w1, gradient_b1, gradient_w0, gradient_b0, learning_rate)  # 更新梯度
            losses.append(avg_loss)  # 记录迭代过程中损失函数值
            # 每进行20此迭代，显示一下当前的损失值
            if (i % 20 == 0):
                print("iter:{},loss:{}".format(i, avg_loss))
        
        return losses, self.w1, self.b1, self.w0, self.b0


# In[30]:


# 画出损失函数随迭代次数增加的情况
def showpeocess(loss, epoch_num):
    plt.title("The Process Of Train")
    plt.plot([i for i in range(epoch_num)], loss)
    plt.xlabel("epoch_num")
    plt.ylabel("loss")
    plt.show()

# 画出测试集标签随迭代次数增加的情况
def shower_true(true, epoch_num):
    plt.title("true")
    plt.plot([i for i in range(epoch_num)], true)
    plt.xlabel("epoch_num")
    plt.ylabel("value")
    plt.show()

# 画出预测值随迭代次数增加的情况
def shower_predict(predict, epoch_num):
    plt.title("predict")
    plt.plot([i for i in range(epoch_num)], predict)
    plt.xlabel("epoch_num")
    plt.ylabel("value")
    plt.show()

# 画出预测值和真实值的对比
def shower(true, predict, epoch_num):
    plt.plot([i for i in range(epoch_num)], true)
    plt.plot([i for i in range(epoch_num)], predict)
    plt.xlabel("num")
    plt.ylabel("value")
    plt.show()


# In[31]:


if __name__ == '__main__':
    Train, Test = deal_data()  # 读取训练集和测试集
    np.random.shuffle(Train)  # 随机打乱
    # 训练数据和标签
    x = Train[:, :-1]
    y = Train[:, -1:]
    # 测试数据和标签
    x1 = Test[:, :-1]
    y1 = Test[:, -1:]
    epoch_num = 1000  # 设置迭代次数
    Model = Model_Config(1, 1)
    losses, w1, b1, w0, b0 = Model.train(epoch_num=epoch_num, x=x, y=y, learning_rate=0.0001)
    predicts = np.dot((np.dot(x1, w0) + b0), w1) + b1  # 测试集的预测值
    showpeocess(loss=losses, epoch_num=epoch_num)  # 画出损失函数随迭代次数的变化
    shower(y1, predicts, 174)  # 画出测试集中真实数据和预测数据之间的对比


# In[24]:





# # paddle

# In[33]:


import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib.pyplot as plt


# In[34]:


BUF_SIZE=500
BATCH_SIZE=20

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


# In[35]:


#用于打印，查看uci_housing数据
train_data=paddle.dataset.uci_housing.train();
sampledata=next(train_data())
print(sampledata)


# In[37]:


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


# In[38]:


cost = fluid.layers.square_error_cost(input=y_predict, label=y) #求一个batch的损失值
avg_cost = fluid.layers.mean(cost)                              #对损失值求平均值


# In[39]:


optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)


# In[40]:


test_program = fluid.default_main_program().clone(for_test=True)


# In[41]:



use_cuda = False                         #use_cuda为False,表示运算场所为CPU;use_cuda为True,表示运算场所为GPU           
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)              #创建一个Executor实例exe
exe.run(fluid.default_startup_program()) #Executor的run()方法执行startup_program(),进行参数初始化


# In[42]:



# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])#feed_list:向模型输入的变量表或变量表名


# In[43]:


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


# In[44]:


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


# In[45]:


infer_exe = fluid.Executor(place)    #创建推测用的executor
inference_scope = fluid.core.Scope() #Scope指定作用域


# In[46]:


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


# In[47]:


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





# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
