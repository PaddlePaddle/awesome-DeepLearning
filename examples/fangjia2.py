import numpy as np
import json


def load_data():
    # 导入数据
    datafile = './housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，前13：影响因素，14：房价中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                     'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # [N,14]
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 划分数据集
    radio = 0.8
    offset = int(data.shape[0] * radio)
    training_data = data[:offset]

    # 计算max，min，avgs
    maximums, minmums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                              training_data.sum(axis=0) / training_data.shape[0]

    # 归一化
    for i in range(feature_num):
        # print(maximums[i],minmums[i],avgs[i])
        data[:, i] = (data[:, i] - minmums[i]) / (maximums[i] - minmums[i])

    # 训练集，测试集
    training_data = data[:offset]
    test_data = data[offset:]

    return training_data, test_data


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        # np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.



    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        z = np.maximum(0,z)
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
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z - y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z - y)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

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
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))

        return losses


# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)