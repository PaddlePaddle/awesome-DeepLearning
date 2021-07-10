# 导入需要用到的package
import numpy as np
import json
import matplotlib.pyplot as plt

def load_data():
    # 从文件导入数据
    datafile = 'housing.data'
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

    # 计算训练集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data



class Network(object):
    def __init__(self, num_of_weights,mid_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, mid_weights)
        self.b = np.zeros(mid_weights)
        self.w2 = np.random.randn(mid_weights,1)
        self.b2 = 0.

    def sigmoid(self,x):
        # sigmoid激活函数
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        #前向传播
        z = np.dot(x, self.w) + self.b
        z = self.sigmoid(z)
        z = np.dot(z,self.w2) + self.b2
        return z

    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

    def gradient(self, x, y):
        # 梯度计算
        z = self.forward(x)
        output1 = self.sigmoid(np.dot(x, self.w) + self.b)
        gradient_w = x.T.dot((z-y).dot(self.w2.T) * ((output1)*(1-output1)))
        gradient_b = np.mean((z-y).dot(self.w2.T) * ((output1)*(1-output1)), axis=0)
        gradient_w2 = np.mean((z-y)*output1,axis=0)
        gradient_w2 = gradient_w2[:,np.newaxis]
        gradient_b2 = np.mean((z-y))
        return gradient_w, gradient_b, gradient_w2, gradient_b2

    def update(self, gradient_w, gradient_w2, gradient_b, gradient_b2, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        self.w2 = self.w2 - eta * gradient_w2
        self.b2 = self.b2 - eta * gradient_b2

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b ,gradient_w2,gradient_b2= self.gradient(x, y)
            self.update(gradient_w, gradient_w2,gradient_b,gradient_b2,eta)
            losses.append(L)
            if (i + 1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses


# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13,2)
num_iterations = 1000
# 启动训练
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()