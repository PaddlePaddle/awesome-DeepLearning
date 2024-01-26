import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # 从文件导入数据
    datafile = 'housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
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
    return training_data, test_data, maximums, minimums, avgs


class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        hid = 13
        self.w1 = np.random.randn(num_of_weights, hid)
        self.b1 = np.zeros(hid)  # 设初始偏置为0
        self.w2 = np.random.rand(hid, 1)
        self.b2 = np.zeros(1)

    def forward(self, x):
        z1 = x.dot(self.w1) + self.b1
        z2 = np.where(z1 < 0, 0, z1)
        z3 = z2.dot(self.w2) + self.b2
        return z3

    def loss(self, y, z3):
        loss = np.mean(np.square(z3-y))
        return loss

    def gradient(self, x, y):
        z1 = x.dot(self.w1) + self.b1
        z2 = np.where(z1 < 0, 0, z1)
        z3 = z2.dot(self.w2) + self.b2
        gradient_y_pred = 2.0 * (z3 - y)
        gradient_w2 = z2.T.dot(gradient_y_pred)
        gradient_temp_relu = gradient_y_pred.dot(self.w2.T)
        gradient_temp_relu[z1 < 0] = 0
        gradient_w1 = x.T.dot(gradient_temp_relu)
        return gradient_w1, gradient_w2

    def update(self, grad_w1, grad_w2, eta=0.01):
        self.w1 -= eta * grad_w1
        self.w2 -= eta * grad_w2

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
                gradient_w1, gradient_w2 = self.gradient(x, y)
                self.update(gradient_w1, gradient_w2, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                      format(epoch_id, iter_id, loss))
        grad_w1 = self.w1
        grad_w2 = self.w2
        return losses, grad_w1, grad_w2


def train():
    # 获取数据
    train_data, test_data, max_values, min_values, avg_values = load_data()
    # 创建网络
    net = Network(13)
    # 启动训练
    losses, grad_w1, grad_w2 = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)
    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()
    # 测试
    idx = np.random.randint(0, test_data.shape[0])
    one_data, label = test_data[idx, :-1], test_data[idx, -1]
    one_data = one_data.reshape([1, -1])
    idx_pred1 = one_data.dot(grad_w1)
    idx_pred2 = np.where(idx_pred1 < 0, 0, idx_pred1)
    idx_pred3 = idx_pred2.dot(grad_w2)
    predict = idx_pred3 * (max_values[-1] - min_values[-1]) + avg_values[-1]
    label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(predict, label))


if __name__ == '__main__':
    train()
