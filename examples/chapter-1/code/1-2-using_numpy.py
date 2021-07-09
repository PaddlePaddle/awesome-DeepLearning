import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    # 从文件导入数据
    datafile = './data/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
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
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), training_data.sum(axis=0) / \
                               training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # np.random.seed(0)
        self.w1 = np.random.randn(num_of_weights, 10)
        self.b1 = np.random.randn(10, 1)
        self.w2 = np.random.randn(10, 1)
        self.b2 = 0.

    def forward(self, x):
        z = np.matmul(x, self.w1) + np.transpose(self.b1)
        z[z < 0] = 0
        z = np.dot(z, self.w2) + self.b2
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        z0 = np.matmul(x, self.w1) + np.transpose(self.b1)
        z0[z0 < 0] = 0
        gradient_w2 = 1. / N * np.sum((z - y) * z0, axis=0)
        gradient_w2 = gradient_w2[:, np.newaxis]
        gradient_b2 = 1. / N * np.sum(z - y)

        z_y = (z - y)
        gradient_w1 = np.zeros((13, 10))
        for i in range(N):
            xi = x[i, :]
            xi = xi[:, np.newaxis]
            gradient_w1 = gradient_w1 + z_y[i] * np.matmul(xi, np.transpose(self.w2))
        gradient_w1 = 1. / N * gradient_w1
        gradient_w1[gradient_w1 < 0] = 0
        gradient_b1 = 1. / N * np.sum((z - y) * np.transpose(self.w2), axis=0)
        gradient_b1[gradient_b1 < 0] = 0
        gradient_b1 = gradient_b1[:, np.newaxis]
        return gradient_w1, gradient_b1, gradient_w2, gradient_b2

    def update(self, gradient_w1, gradient_b1, gradient_w2, gradient_b2, eta=0.01):
        self.w1 = self.w1 - eta * gradient_w1
        self.b1 = self.b1 - eta * gradient_b1
        self.w2 = self.w2 - eta * gradient_w2
        self.b2 = self.b2 - eta * gradient_b2

    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                # print(self.w.shape)
                # print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w1, gradient_b1, gradient_w2, gradient_b2 = self.gradient(x, y)
                self.update(gradient_w1, gradient_b1, gradient_w2, gradient_b2, eta)
                losses.append(loss)
            print('Epoch {:3d} , loss = {:.4f}'.
                  format(epoch_id, loss))

        return losses


def train():
    # 获取数据
    train_data, test_data = load_data()

    # 创建网络
    net = Network(13)
    # 启动训练
    losses = net.train(train_data, num_epochs=10, batch_size=50, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()

if __name__ == '__main__':
    train()