"""
The implementation of neural network using numpy.
"""
# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    # 对数据进行归一化处理
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
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i + 1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses


def train():
    # 获取数据
    train_data, test_data = load_data()
    x = train_data[:, :-1]
    y = train_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations = 1000
    # 启动训练
    losses = net.train(x, y, iterations=num_iterations, eta=0.01)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()

def plot_3D_neural_work_weiight():
    # 获取数据
    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = Network(13)
    losses = []
    # 只画出参数w5和w9在区间[-160, 160]的曲线部分，已经包含损失函数的极值
    w5 = np.arange(-160.0, 160.0, 1.0)
    w9 = np.arange(-160.0, 160.0, 1.0)
    losses = np.zeros([len(w5), len(w9)])

    # 计算设定区域内每个参数取值所对应的Loss
    for i in range(len(w5)):
        for j in range(len(w9)):
            net.w[5] = w5[i]
            net.w[9] = w9[j]
            z = net.forward(x)
            loss = net.loss(z, y)
            losses[i, j] = loss

    # 将两个变量和对应的Loss作3D图
    fig = plt.figure()
    ax = Axes3D(fig)

    w5, w9 = np.meshgrid(w5, w9)

    ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
    plt.show()


if __name__ == '__main__':

    plot_3D_neural_work_weiight()

    train()