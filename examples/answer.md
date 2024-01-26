1.深度学习发展历史：
1.1第一代神经网络（1958-1969）
最早的神经网络的思想起源于1943年的MCP人工神经元模型，当时是希望能够用计算机来模拟人的神经元反应的过程，该模型将神经元简化为了三个过程：输入信号线性加权，求和，非线性激活（阈值法）。如下图所示
![](https://ai-studio-static-online.cdn.bcebos.com/4ba1cbdf71d64b4d8c4978dfa557b28a07c4fd0479094db59ec40b8420672d7d)

第一次将MCP用于机器学习（分类）的当属1958年Rosenblatt发明的感知器（perceptron）算法。该算法使用MCP模型对输入的多维数据进行二分类，且能够使用梯度下降法从训练样本中自动学习更新权值。1962年，该方法被证明为能够收敛，理论与实践效果引起第一次神经网络的浪潮。
然而学科发展的历史不总是一帆风顺的。
1969年，美国数学家及人工智能先驱Minsky在其著作中证明了感知器本质上是一种线性模型，只能处理线性分类问题，就连最简单的XOR（亦或）问题都无法正确分类。这等于直接宣判了感知器的死刑，神经网络的研究也陷入了近20年的停滞。
1.2第二代神经网络（1986-1998）
第一次打破非线性诅咒的当属现代DL大牛Hinton，其在1986年发明了适用于多层感知器（MLP）的BP算法，并采用Sigmoid进行非线性映射，有效解决了非线性分类和学习的问题。该方法引起了神经网络的第二次热潮。
1989年，Robert Hecht-Nielsen证明了MLP的万能逼近定理，即对于任何闭区间内的一个连续函数f，都可以用含有一个隐含层的BP网络来逼近该定理的发现极大的鼓舞了神经网络的研究人员。
也是在1989年，LeCun发明了卷积神经网络-LeNet，并将其用于数字识别，且取得了较好的成绩，不过当时并没有引起足够的注意。
值得强调的是在1989年以后由于没有特别突出的方法被提出，且NN一直缺少相应的严格的数学理论支持，神经网络的热潮渐渐冷淡下去。冰点来自于1991年，BP算法被指出存在梯度消失问题，即在误差梯度后向传递的过程中，后层梯度以乘性方式叠加到前层，由于Sigmoid函数的饱和特性，后层梯度本来就小，误差梯度传到前层时几乎为0，因此无法对前层进行有效的学习，该发现对此时的NN发展雪上加霜。
1997年，LSTM模型被发明，尽管该模型在序列建模上的特性非常突出，但由于正处于NN的下坡期，也没有引起足够的重视。
1.3统计学习方法的春天（1986-2006）
1986年，决策树方法被提出，很快ID3，ID4，CART等改进的决策树方法相继出现，到目前仍然是非常常用的一种机器学习方法。该方法也是符号学习方法的代表。 
1995年，线性SVM被统计学家Vapnik提出。该方法的特点有两个：由非常完美的数学理论推导而来（统计学与凸优化等），符合人的直观感受（最大间隔）。不过，最重要的还是该方法在线性分类的问题上取得了当时最好的成绩。 
1997年，AdaBoost被提出，该方法是PAC（Probably Approximately Correct）理论在机器学习实践上的代表，也催生了集成方法这一类。该方法通过一系列的弱分类器集成，达到强分类器的效果。 
2000年，KernelSVM被提出，核化的SVM通过一种巧妙的方式将原空间线性不可分的问题，通过Kernel映射成高维空间的线性可分问题，成功解决了非线性分类的问题，且分类效果非常好。至此也更加终结了NN时代。 
2001年，随机森林被提出，这是集成方法的另一代表，该方法的理论扎实，比AdaBoost更好的抑制过拟合问题，实际效果也非常不错。 
2001年，一种新的统一框架-图模型被提出，该方法试图统一机器学习混乱的方法，如朴素贝叶斯，SVM，隐马尔可夫模型等，为各种学习方法提供一个统一的描述框架。
1.4第三代神经网络-DL（2006-至今）
2006年，DL元年。是年，Hinton提出了深层网络训练中梯度消失问题的解决方案：无监督预训练对权值进行初始化+有监督训练微调。其主要思想是先通过自学习的方法学习到训练数据的结构（自动编码器），然后在该结构上进行有监督训练微调。但是由于没有特别有效的实验验证，该论文并没有引起重视。
2011年，ReLU激活函数被提出，该激活函数能够有效的抑制梯度消失问题。
2011年，微软首次将DL应用在语音识别上，取得了重大突破。
2012年，Hinton课题组为了证明深度学习的潜力，首次参加ImageNet图像识别比赛，其通过构建的CNN网络AlexNet一举夺得冠军，且碾压第二名（SVM方法）的分类性能。也正是由于该比赛，CNN吸引到了众多研究者的注意。 
AlexNet的创新点： 
（1）首次采用ReLU激活函数，极大增大收敛速度且从根本上解决了梯度消失问题；（2）由于ReLU方法可以很好抑制梯度消失问题，AlexNet抛弃了“预训练+微调”的方法，完全采用有监督训练。也正因为如此，DL的主流学习方法也因此变为了纯粹的有监督学习；（3）扩展了LeNet5结构，添加Dropout层减小过拟合，LRN层增强泛化能力/减小过拟合；（4）首次采用GPU对计算进行加速；

2013,2014,2015年，通过ImageNet图像识别比赛，DL的网络结构，训练方法，GPU硬件的不断进步，促使其在其他领域也在不断的征服战场
2015年，Hinton，LeCun，Bengio论证了局部极值问题对于DL的影响，结果是Loss的局部极值问题对于深层网络来说影响可以忽略。该论断也消除了笼罩在神经网络上的局部极值问题的阴霾。具体原因是深层网络虽然局部极值非常多，但是通过DL的BatchGradientDescent优化方法很难陷进去，而且就算陷进去，其局部极小值点与全局极小值点也是非常接近，但是浅层网络却不然，其拥有较少的局部极小值点，但是却很容易陷进去，且这些局部极小值点与全局极小值点相差较大。论述原文其实没有证明，只是简单叙述，严密论证是猜的。。。
2015，DeepResidualNet发明。分层预训练，ReLU和BatchNormalization都是为了解决深度神经网络优化时的梯度消失或者爆炸问题。但是在对更深层的神经网络进行优化时，又出现了新的Degradation问题，即”通常来说，如果在VGG16后面加上若干个单位映射，网络的输出特性将和VGG16一样，这说明更深次的网络其潜在的分类性能只可能>=VGG16的性能，不可能变坏，然而实际效果却是只是简单的加深VGG16的话，分类性能会下降（不考虑模型过拟合问题）“Residual网络认为这说明DL网络在学习单位映射方面有困难，因此设计了一个对于单位映射（或接近单位映射）有较强学习能力的DL网络，极大的增强了DL网络的表达能力。此方法能够轻松的训练高达150层的网络。

2.人工智能、机器学习、深度学习的区别与联系：
 2.1人工智能（Artificial Intelligence）
人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的一门技术科学。“人工智能”是“一门技术科学”，它研究与开发的对象是“理论、技术及应用系统”，研究的目的是为了“模拟、延伸和扩展人的智能”。人工智能在50年代就提出了。
 2.2 机器学习   
随着人对计算机科学的期望越来越高，要求它解决的问题越来越复杂，已经远远不能满足人们的诉求了。于是有人提出了一个新的思路——能否不为难研究者，让机器自己去学习呢？机器学习就是用算法解析数据，不断学习，对世界中发生的事做出判断和预测的一项技术。研究人员不会亲手编写软件、确定特殊指令集、然后让程序完成特殊任务；相反，研究人员会用大量数据和算法“训练”机器，让机器学会如何执行任务。这里有三个重要的信息：
- 1、“机器学习”是“模拟、延伸和扩展人的智能”的一条路径，所以是人工智能的一个子集；
- 2、“机器学习”是要基于大量数据的，也就是说它的“智能”是用大量数据喂出来的；
- 3、正是因为要处理海量数据，所以大数据技术尤为重要；“机器学习”只是大数据技术上的一个应用。常用的10大机器学习算法有：决策树、随机森林、逻辑回归、SVM、朴素贝叶斯、K最近邻算法、K均值算法、Adaboost算法、神经网络、马尔科夫。
 2.3 深度学习
相较而言，深度学习是一个比较新的概念，严格地说是2006年提出的。深度学习是用于建立、模拟人脑进行分析学习的神经网络，并模仿人脑的机制来解释数据的一种机器学习技术。它的基本特点，是试图模仿大脑的神经元之间传递，处理信息的模式。最显著的应用是计算机视觉和自然语言处理(NLP)领域。显然，“深度学习”是与机器学习中的“神经网络”是强相关，“神经网络”也是其主要的算法和手段；或者我们可以将“深度学习”称之为“改良版的神经网络”算法。深度学习又分为卷积神经网络（Convolutional neural networks，简称CNN）和深度置信网（Deep Belief Nets，简称DBN）。其主要的思想就是模拟人的神经元，每个神经元接受到信息，处理完后传递给与之相邻的所有神经元即可。
联系：深度学习是一种机器学习技术，机器学习是人工智能研究中的重要分支。

3.神经元、单层感知机、多层感知机：
3.1神经元
即神经元细胞，是神经系统最基本的结构和功能单位。分为细胞体和突起两部分。细胞体由细胞核、细胞膜、细胞质组成，具有联络和整合输入信息并传出信息的作用。突起有树突和轴突两种。树突短而分枝多，直接由细胞体扩张突出，形成树枝状，其作用是接受其他神经元轴突传来的冲动并传给细胞体。轴突长而分枝少，为粗细均匀的细长突起，常起于轴丘，其作用是接受外来刺激，再由细胞体传出。轴突除分出侧枝外，其末端形成树枝样的神经末梢。末梢分布于某些组织器官内，形成各种神经末梢装置。感觉神经末梢形成各种感受器；运动神经末梢分布于骨骼肌肉，形成运动终极。
3.2单层感知机
单层感知机是二分类的线性分类模型，输入是被感知数据集的特征向量，输出时数据集的类别{+1,-1}。单层感知机的函数近似非常有限，其决策边界必须是一个超平面，严格要求数据是线性可分的。
3.3多层感知机
多层感知机（MLP，Multilayer Perceptron）也叫人工神经网络（ANN，Artificial Neural Network），除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构.。

4.什么是前向传播？
所谓前向传播，就是给网络输入一个样本向量，该样本向量的各元素，经过各隐藏层的逐级加权求和+非线性激活，最终由输出层输出一个预测向量的过程。
如下图，图右上角是f(x,y,z)=(x+y)*z 的计算tu
![](https://ai-studio-static-online.cdn.bcebos.com/ee2da10825f24ad7be7be8398bd58a427ec049006e18463db0f33f1ab0fe3fc6)
分别赋值 x = − 2 ， y = 5 ， z = − 4 x = -2，y = 5， z = -4x=−2，y=5，z=−4，从计算图的左边开始，数据开始流动，依次计算出 q 、 f q、fq、f。

最终得到计算图中那 6 个绿色的数字，这就是前向传播的结果。

5.什么是反向传播？
按照我的拙见，反向传播是损失函数对权重的偏导数，为了求出来具体的数值，用了函数求导时候的链式法则。根据求出来的导数值大小来看权重大小对损失函数的影响，方便下一步进行权重更新，进而最小化损失函数，求出较为合适的权重，即找到在选择的模型空间下的一个较为优的解！
![](https://ai-studio-static-online.cdn.bcebos.com/c196204c7d0c41b5845fc653a803917eb35b5f9ff40740858f76869c6e484ce7)
![](https://ai-studio-static-online.cdn.bcebos.com/8f25a27338444458870f7029f6cb5ee90a6bc573b3da4f3fad4fdcb0c0e85aaf)


房价预测python+numpy实现


```python
from sklearn.datasets import load_boston 
import numpy as np
import matplotlib.pyplot as plt
def load_data():   
    datafile = './data/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 2.将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 3.将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 4.对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)       
        self.n_hidden = 10
        self.w1 = np.random.randn(num_of_weights,10)  # 设置随机的权重
        self.b1 = np.zeros(10)  # 这里偏置为0
        self.w2 = np.random.rand(10,1)  # 这里因为输出只有一个模型，所以输出维度为1
        self.b2 = np.zeros(1)
                   
    def Relu(self,x):
        return np.where(x < 0,0,x)
    
    def MSE_loss(self, y,y_pred):
        return np.mean(np.square(y_pred - y))
        
    def Linear(self,x,w,b):
        z = x.dot(w) + b
        return z
    
    def back_gradient(self, y_pred, y,s1,l1):
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = s1.T.dot(grad_y_pred)
        grad_temp_relu = grad_y_pred.dot(self.w2.T)
        grad_temp_relu[l1 < 0] = 0
        grad_w1 = x.T.dot(grad_temp_relu) 
        return grad_w1, grad_w2
    
    def update(self, grad_w1,grad_w2,learning_rate):
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2      
            
    def train(self, x, y, iterations, learning_rate):
        losses = []  # 记录每次迭代损失值
        for t in range(num_iterations):
            # 前向传播
            l1 = self.Linear(x,self.w1,self.b1)
            s1 = self.Relu(l1)
            y_pred = self.Linear(s1,self.w2,self.b2)
            # 计算损失函数
            loss = self.MSE_loss(y,y_pred)
            losses.append(loss)
            # 反向传播
            grad_w1,grad_w2 = self.back_gradient(y_pred, y,s1,l1)
            # 权重更新
            self.update(grad_w1,grad_w2,learning_rate)          
        return losses
        
# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]

# 创建网络
net = Network(13)
num_iterations=50000
# 启动训练
losses = net.train(x,y, iterations = num_iterations, learning_rate = 1e-6)

# 画出损失函数的变化趋势    
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(plot_x, plot_y)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.show()
#print('w1 = {}\n w2 = {}'.format(w1,w2))


```


![png](output_6_0.png)


paddle实现房价预测


```python
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random

#数据处理
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
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
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    #ratio = 0.8
    #offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

#模型设计
class Regressor(fluid.dygraph.Layer):
    #声明每一层网络的实现函数
    def __init__(self):
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=13, output_dim=1, act=None)
    
    # 构建神经网络结构，实现前向计算过程，并返回预测结果
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

#训练配置
# 定义飞桨动态图的工作环境
with fluid.dygraph.guard():
    # 声明定义好的线性回归模型
    model = Regressor()
    # 开启模型训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data()
    # 定义优化算法，这里使用随机梯度下降-SGD
    # 学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())

#训练过程
with dygraph.guard(fluid.CPUPlace()):
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
            x = np.array(mini_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)
            
            # 前向计算
            predicts = model(house_features)
            
            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
    # 保存模型
    fluid.save_dygraph(model.state_dict(), 'LR_model')

#保存模型
# 定义飞桨动态图工作环境
with fluid.dygraph.guard():
    # 保存模型参数，文件名为LR_model
    fluid.save_dygraph(model.state_dict(), 'LR_model')
    print("模型保存成功，模型参数保存在LR_model中")

#测试模型
def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # 选择倒数第10条数据用于测试
    tmp = datas[-10]
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]

    # 对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] - min_values[i])

    data = np.reshape(np.array(one_data[:-1]), [1, -1]).astype(np.float32)
    label = one_data[-1]
    return data, label

with dygraph.guard():
    # 参数为保存模型参数的文件地址
    model_dict, _ = fluid.load_dygraph('LR_model')
    model.load_dict(model_dict)
    model.eval()

    # 参数为数据集的文件地址
    test_data, label = load_one_example('./work/housing.data')
    # 将数据转为动态图的variable格式
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)

    # 对结果做反归一化处理
    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(results.numpy(), label))

```

    epoch: 0, iter: 0, loss is: [0.06904466]
    epoch: 0, iter: 20, loss is: [0.0605798]
    epoch: 0, iter: 40, loss is: [0.08289625]
    epoch: 1, iter: 0, loss is: [0.08934028]
    epoch: 1, iter: 20, loss is: [0.06063873]
    epoch: 1, iter: 40, loss is: [0.04450267]
    epoch: 2, iter: 0, loss is: [0.06582879]
    epoch: 2, iter: 20, loss is: [0.04792718]
    epoch: 2, iter: 40, loss is: [0.09931071]
    epoch: 3, iter: 0, loss is: [0.09746288]
    epoch: 3, iter: 20, loss is: [0.0253105]
    epoch: 3, iter: 40, loss is: [0.1328317]
    epoch: 4, iter: 0, loss is: [0.02627663]
    epoch: 4, iter: 20, loss is: [0.03310246]
    epoch: 4, iter: 40, loss is: [0.04306072]
    epoch: 5, iter: 0, loss is: [0.02750561]
    epoch: 5, iter: 20, loss is: [0.04128321]
    epoch: 5, iter: 40, loss is: [0.13451238]
    epoch: 6, iter: 0, loss is: [0.02393558]
    epoch: 6, iter: 20, loss is: [0.02832639]
    epoch: 6, iter: 40, loss is: [0.03099384]
    epoch: 7, iter: 0, loss is: [0.06580763]
    epoch: 7, iter: 20, loss is: [0.04271714]
    epoch: 7, iter: 40, loss is: [0.0485788]
    epoch: 8, iter: 0, loss is: [0.04531223]
    epoch: 8, iter: 20, loss is: [0.07685316]
    epoch: 8, iter: 40, loss is: [0.02358524]
    epoch: 9, iter: 0, loss is: [0.07324368]
    epoch: 9, iter: 20, loss is: [0.05917173]
    epoch: 9, iter: 40, loss is: [0.0629088]
    模型保存成功，模型参数保存在LR_model中
    Inference result is [[15.3298645]], the corresponding label is 19.7


通过对比，不难看出，numpy由于训练次数较多，损失函数最终较小；
两者预测结果对比，大致相同。
