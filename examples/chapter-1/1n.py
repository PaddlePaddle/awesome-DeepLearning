import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.function_base import gradient

def load_data():
    datafile = 'E:/VSCODE/baidu/awesome-DeepLearning-master/junior_class/chapter-1-hands_on_deep_learning/code/data/housing.data'
    data = np.fromfile(datafile, sep=' ')

    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    data = data.reshape([data.shape[0] // feature_num, feature_num])

    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    maximums, minimums, avgs = training_data.max(axis = 0), training_data.min(axis = 0), training_data.sum(axis = 0) / training_data.shape[0]
    std = training_data.std(axis = 0)
#数据归一化
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / std[i]
        #data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
        
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data

class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(0) 
        self.w1 = np.random.randn(num_of_weights, 10)
        self.b1 = np.zeros(10)
        self.w2 = np.random.randn(10,1)
        self.b2 = np.zeros(1)
    
    def Relu(self,x):
        return np.where(x < 0,0,x)

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        z1 = self.Relu(z1)
        z = np.dot(z1, self.w2) + self.b2
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
        gradient_w1 = 1. / N * np.sum((z - y) * x, axis = 0)
        gradient_w1 = gradient_w1[:, np.newaxis]
        gradient_b1 = 1. / N * np.sum(z - y)

        
        return gradient_w1, gradient_b1

    def updata(self, gradient_w1, gradient_b1, eta = 0.01):
        self.w1 = self.w1 + eta * gradient_w1
        self.b1 = self.b1 + eta * gradient_b1


    def train(self, training_data, num_epochs, batch_size = 10, eta = 0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w1, gradient_b1 = self.gradient(x, y)
                self.updata(gradient_w1, gradient_b1, eta)
                losses.append(loss)
                print('epoch{:3d} / iter{:3d}, loss = {:.4f}'.format(epoch_id, iter_id, loss))
        return losses

def train():
    train_data, test_data = load_data()
    net = Network(13)
    losses = net.train(train_data, num_epochs = 50, batch_size = 100, eta = 0.1)
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()

def plot_3D_neural_work_weight():

    training_data, test_data = load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]

    net = Network(13)
    losses = []
    w5 = np.arange(-160.0, 160.0, 1.0)
    w9 = np.arange(-160.0, 160.0, 1.0)
    losses = np.zeros([len(w5), len(w9)])

    for i in range(len(w5)):
        for j in range(len(w9)):
            net.w1[5] = w5[i]
            net.w1[9] = w9[j]
            z = net.forward(x)
            loss = net.loss(z, y)
            losses[i, j] = loss

    fig = plt.figure()
    ax = Axes3D(fig)

    w5, w9 = np.meshgrid(w5, w9)

    ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

if __name__ == '__main__':
    plot_3D_neural_work_weight()
    train()
    
                

