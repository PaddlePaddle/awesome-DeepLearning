import numpy as np
import matplotlib.pyplot as plt

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

    # 计算训练集的最大值，最小值，平均值，axis=0即按行计算，寻找样本集中每一列特征的最值|按行求和
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        #在这里是一列一列依次计算
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # 两层全连接层——————13*20*1
        np.random.seed(0)
        self.n_hidden = 20
        self.w1 = np.random.randn(num_of_weights, self.n_hidden)  # 设置随机的权重
        self.b1 = np.zeros(self.n_hidden)  # 这里偏置为0
        self.w2 = np.random.rand(self.n_hidden, 1)  # 这里因为输出只有一个模型，所以输出维度为1
        self.b2 = np.zeros(1)

    def Relu(self, x):
        return np.where(x < 0, 0, x)

    def MSE_loss(self, y, y_pred):
        return np.mean(np.square(y_pred - y))

    def Linear(self, x, w, b):
        z = x.dot(w) + b
        return z

    def back_gradient(self, y_pred, y, s1, a1):
        # a1=w1x1+b1,s1=relu(a1),y_pred=w2s1+b2
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = s1.T.dot(grad_y_pred)
        grad_s1 = grad_y_pred.dot(self.w2.T)
        grad_s1[a1 < 0] = 0
        grad_w1 = x.T.dot(grad_s1)
        # 这里没有管b1和b2
        return grad_w1, grad_w2

    def update(self, grad_w1, grad_w2, learning_rate):
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2

    def train(self, x, y, iterations, learning_rate):
        losses = []  # 记录每次迭代损失值
        for t in range(num_iterations):
            # 前向传播
            a1 = self.Linear(x, self.w1, self.b1)
            s1 = self.Relu(a1)
            y_pred = self.Linear(s1, self.w2, self.b2)
            # 计算损失函数
            loss = self.MSE_loss(y, y_pred)
            losses.append(loss)
            # 反向传播
            grad_w1, grad_w2 = self.back_gradient(y_pred, y, s1, a1)
            # 权重更新
            self.update(grad_w1, grad_w2, learning_rate)
        return losses


# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]

# 创建网络
net = Network(13)
num_iterations = 500
# 启动训练
losses = net.train(x, y, iterations=num_iterations, learning_rate=1e-6)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(plot_x, plot_y)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()
print('w1 = {}\n w2 = {}'.format(net.w1, net.w2))



#加载飞桨、Numpy和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random


class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc1 = Linear(in_features=13, out_features=20)
        self.fc2 = Linear(in_features=20, out_features=1)

    # 网络的前向计算
    def forward(self, inputs):
        relu = paddle.nn.ReLU()  # 不这样引用relu会报错
        # p_outputs1 = self.fc1(inputs)
        p_outputs1 = relu(self.fc1(inputs))
        p_outputs2 = self.fc2(p_outputs1)
        return p_outputs2

# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 加载数据
training_data, test_data = load_data()
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 20  # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小
pd_losses = []
# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:])  # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算
        predicts = model(house_features)
        # print (predicts)
        # 计算损失
        loss = F.square_error_cost(paddle.to_tensor(predicts), label=prices)
        avg_loss = paddle.mean(loss)

        if iter_id % 20 == 0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            pd_losses.append(avg_loss.numpy())

        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()

import numpy as np
import matplotlib.pyplot as plt
# 画出损失函数的变化趋势
plot_px = np.arange(len(pd_losses))
plot_py = np.array(pd_losses)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(plot_px, plot_py)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()