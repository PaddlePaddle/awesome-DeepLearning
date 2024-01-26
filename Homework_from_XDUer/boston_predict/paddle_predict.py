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
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0] // feature_num, feature_num])

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                               training_data.sum(axis=0) / training_data.shape[0]

    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    training_data = data[:offset]
    testing_data = data[offset:]
    return training_data, testing_data


class model(paddle.nn.Layer):

    def __init__(self):
        super(model, self).__init__()

        self.fc1 = Linear(in_features=13, out_features=13)  # 13 * 13
        self.fc2 = Linear(in_features=13, out_features=1)  # 13 * 1

    def forward(self, inputs):
        x = F.sigmoid(self.fc1(inputs))
        x = self.fc2(x)
        return x


def train(epochs, batch_size):
    losses = []
    for epoch in range(epochs):
        np.random.shuffle(training_data)
        batches = [training_data[k : k + batch_size] for k in range(0, len(training_data), batch_size)]
        sum_loss = 0.0
        for iter, mini_batch in enumerate(batches):
            x = np.array(mini_batch[:, :-1])
            y = np.array(mini_batch[:, -1:])
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)

            predicts = model(x)

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
    x = paddle.to_tensor(x)
    y = paddle.to_tensor(y)

    predicts = model(x)

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

    training_data, testing_data = load_data()

    model = model()
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())

    epochs = 200
    batch_size = 100

    losses = train(epochs, batch_size)

    plotLoss(epochs, losses)

    testing()

