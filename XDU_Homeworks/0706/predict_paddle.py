# !usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import paddle
import numpy as np
from paddle.nn import Linear
import matplotlib.pyplot as plt
import paddle.nn.functional as F


def load_data():
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)
    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0] // feature_num, feature_num])

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                               training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    # 训练集和测试集的划分比例
    training_data = data[:offset]
    testing_data = data[offset:]
    return training_data, testing_data


class model(paddle.nn.Layer):

    def __init__(self):
        # 初始化父类中的一些参数
        super(model, self).__init__()

        # 定义两层全连接层
        self.fc1 = Linear(in_features=13, out_features=13)  # 13 * 13
        self.fc2 = Linear(in_features=13, out_features=1)  # 13 * 1

    # 网络的前向计算
    def forward(self, inputs):
        x = F.sigmoid(self.fc1(inputs))
        x = self.fc2(x)
        return x


def train(epochs, batch_size):
    # 定义外层循环
    losses = []
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        batches = [training_data[k : k + batch_size] for k in range(0, len(training_data), batch_size)]
        sum_loss = 0.0
        for iter, mini_batch in enumerate(batches):
            x = np.array(mini_batch[:, :-1])  # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:])  # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图tensor形式
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)

            # forward传播
            predicts = model(x)

            # loss计算
            loss = F.square_error_cost(predicts, label=y)
            avg_loss = paddle.mean(loss)
            sum_loss += avg_loss
            if iter % 20 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch, iter, np.squeeze(avg_loss.numpy())))

            avg_loss.backward() # BP计算梯度
            opt.step() # 更新参数
            opt.clear_grad() # 梯度清零
        losses.append(sum_loss / len(batches))

    return losses


def testing():
    x = np.array(testing_data[:, :-1])  # 获得当前批次训练数据
    y = np.array(testing_data[:, -1:])  # 获得当前批次训练标签（真实房价）
    # 将numpy数据转为飞桨动态图tensor形式
    x = paddle.to_tensor(x)
    y = paddle.to_tensor(y)

    # forward传播
    predicts = model(x)

    # loss计算
    loss = F.square_error_cost(predicts, label=y)
    avg_loss = paddle.mean(loss)
    print("Testing Loss: {}".format(np.squeeze(avg_loss.numpy())))


def plotLoss(epoch, loss):
    plt.title("Loss Curve in Training")
    plt.plot([i for i in range(epoch)], loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()



if __name__ == '__main__':

    # 加载数据
    training_data, testing_data = load_data()

    # 加载模型
    model = model()
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters()) # optimizer采用SGD, lr设置与之前一样

    epochs = 200
    batch_size = 100

    losses = train(epochs, batch_size)

    plotLoss(epochs, losses)

    testing()

