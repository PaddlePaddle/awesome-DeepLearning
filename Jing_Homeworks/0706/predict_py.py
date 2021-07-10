# !usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import matplotlib.pyplot as plt

def load_data():
    # 读入训练数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 数据形状变换: 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num]) #506 * 14

    # 数据集划分
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset] #404*14

    # 特征值归一化
    maximums, minimums, avgs = \
                         training_data.max(axis=0), \
                         training_data.min(axis=0), \
         training_data.sum(axis=0) / training_data.shape[0]
    # 对数据进行归一化处理,计算训练集的最大值，最小值，平均值
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    training_data = data[:offset]
    testing_data = data[offset:]
    np.random.shuffle(training_data)

    return training_data, testing_data

def sigmoid(x):
    #sigmoid 激活函数
    return 1 / (1 + np.exp(-x))

def dsig(x):
    # sigmoid 函数导数
    return x * (1 - x)

class model(object):
    def __init__(self, nums_of_weight, nums_of_hidden):
        np.random.seed(233)
        self.w0 = np.random.randn(nums_of_weight, nums_of_hidden)
        self.b0 = np.zeros(nums_of_hidden)
        self.w1 = np.random.randn(nums_of_hidden, 1)
        self.b1 = np.zeros(1)

    # 前向传播
    def forward(self, x):
        z = np.dot(x, self.w0) + self.b0
        z = sigmoid(z)
        output = np.dot(z, self.w1) + self.b1
        return z, output

    def loss(self,z, y):
        err = z - y
        cost = np.mean(err * err) / 2
        return cost

    # 计算梯度(Core BP)
    def back(self, x, y):
        z, output = self.forward(x)

        # 第一个全连接层参数的梯度
        gradient_w0 = x.T.dot((output-y).dot(self.w1.T) * dsig(z))
        gradient_b0 = np.mean((output-y).dot(self.w1.T) * dsig(z), axis=0)

        # 第二个全连接层参数的梯度
        gradient_w1 = np.mean((output-y)*z,axis=0)
        gradient_w1 = gradient_w1[:,np.newaxis]
        gradient_b1 = np.mean((output-y))

        return gradient_w1, gradient_b1, gradient_w0, gradient_b0

    # 使用梯度更新权值参数w
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
            _, z = self.forward(x)
            avg_loss = self.loss(z, y)
            gradient_w1, gradient_b1, gradient_w0, gradient_b0 = self.back(x, y)
            self.update(gradient_w1, gradient_b1, gradient_w0, gradient_b0, learning_rate)
            losses.append(avg_loss)
            if (i % 10 == 0):
                print("iter:{},loss:{}".format(i, avg_loss))

        return losses


def plotLoss(epoch, loss):
    plt.title("Loss Curve in Training")
    plt.plot([i for i in range(epoch)], loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def main():
    # 获取数据
    training_data, test_data = load_data()

    #只获取前13列特征
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = model(13, 13)
    epoch = 200
    loss = net.train(epoch_num=epoch, x=x, y=y, learning_rate=1e-3)

    plotLoss(epoch, loss)



if __name__ == '__main__':
    main()


