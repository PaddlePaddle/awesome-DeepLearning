import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data():
    # 从文件导入数据
    datafile ='D:/tools/test data/1/housing.data '
    data = np.fromfile(datafile, sep=' ')
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(
        axis=0), training_data.sum(axis=0) / training_data.shape[0]

    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

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
                print("iter {}, loss {}".format(i, L))
        return losses


# 加载数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]

# 创建网络
net = Network(13)
num_iterations = 1000
# 训练
losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()

