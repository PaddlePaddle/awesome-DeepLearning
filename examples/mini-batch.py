import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 导入房价数据
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')
    # 将原始数据Reshape 并且拆分成训练集和测试集
    data = data.reshape([-1, 14])
    offset = int(data.shape[0]*0.8)
    train_data = data[:offset]
    # 归一化处理
    maximums, minimums, avgs = train_data.max(axis=0), train_data.min(axis=0), train_data.sum(axis=0) / train_data.shape[0]
    for i in range(14):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    train_data = data[:offset]
    test_data = data[offset:]
    return train_data, test_data


class Network(object):
    def __init__(self, num_of_weight):
        # 随机产生w初始值
        # np.random.seed(0)
        # randn函数返回一组样本，具有标准正态分布，维度为[num_of_weight, 1]
        self.w = np.random.randn(num_of_weight, 1)
        self.b = 0.

    def forword(self, x):   # 前向计算
        z = np.dot(x, self.w) + self.b
        return z

    def loss(self, z, y):   # loss计算
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost

    def gradient(self, x, y):   # 计算梯度
        z = self.forword(x)
        gradient_w = np.mean((z - y)*x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]  # [13,] -> [13, 1]
        gradient_b = np.mean((z - y), axis=0)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta=0.01):     # 更新参数
        self.w = self.w - eta*gradient_w
        self.b = self.b - eta*gradient_b

    def train(self, train_data, num_epcches, batch_size=10, eta=0.01):    # 训练代码
        n = len(train_data)
        print(n)
        losses = []
        for epoch_id in range(num_epcches):
            # 在每轮迭代之前，将训练数据的顺序随机打乱
            np.random.shuffle(train_data)
            # 将数据拆分，每个mini_batch包含batch_size条数据
            mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
            # enumerate()函数将一个可遍历的数据对象组合为一个索引列表，同时列出数据下标和数据
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                z = self.forword(x)
                loss = self.loss(z, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch{:3d} / iter {:3d}, loss = {:.4f}'.format(epoch_id, iter_id, loss))
        return losses


train_data, test_data = load_data()
net = Network(13)
# 开始训练
losses = net.train(train_data, batch_size=50, num_epcches=100, eta=0.01)
# 画出损失函数变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
