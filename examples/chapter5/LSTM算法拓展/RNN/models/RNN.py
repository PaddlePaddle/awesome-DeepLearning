import sys
import numpy as np
from utils import *
from datetime import datetime


class RNN:
    def __init__(self, word_dim = 40, hidden_dim = 100, bptt_truncate=4):
        
        self.word_dim = word_dim #单词维度
        self.hidden_dim = hidden_dim # 隐藏层数量
        self.bptt_truncate = bptt_truncate
        # 输入权重矩阵 H*K 
        self.U = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
        # 隐藏层权重矩阵 H*H
        self.V = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (word_dim, hidden_dim))
        # 输出层权重矩阵 K*H
        self.W = np.random.uniform(-np.sqrt(1.0/hidden_dim), np.sqrt(1.0/hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        '''
        进行前向传播
        Args:
            x: 输入句子序列，一个句子
        Return:
            h: 各时刻隐藏层输出
            o：各时刻softmax层的输出

        '''
        # 句子长度
        T = len(x)
        # 隐藏层各时刻输出
        s = np.zeros((T+1, self.hidden_dim))
        # 各时刻实际输出
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.W.dot(s[t-1]) + self.U[:,x[t]])
            o[t] = softmax(self.V.dot(s[t]))
        return [o,s]

    def predict(self, x):
        '''
        对x进行前向传播后选择得分最高的index
        Args:
            x: 输入句子序列，一个句子
        Return:index

        '''
        o,s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calc_total_loss(self, X_train, Y_train):
        '''
        计算交叉熵损失
        Args:
            X_train: 训练集
            Y_train: 训练集标签
        Return:
            E:训练集上的交叉熵
        '''
        E = 0
        for x,y in zip(X_train, Y_train):
            # 对每个样例
            o, s = self.forward_propagation(x)
            # 取出对每个word后的正确word的估计
            correct_word_predictions = o[np.arange(len(y)),y]
            # 计算熵
            E += -1 * np.sum(np.log(correct_word_predictions))
        return E

    def calc_avg_loss(self, X_train, Y_train):
        '''
        计算每个单词上的平均交叉熵
        '''
        num_total_words = np.sum((len(y_i) for y_i in Y_train))
        return self.calc_total_loss(X_train, Y_train) / num_total_words

    def bptt(self, x, y):
        '''
        反向传播
        Args:
            x: 输入句子序列，一个句子
            y：输出句子序列
        Return:
            dLdU: U的梯度
            dLdV: V的梯度
            dLdW: W的梯度

        '''
        T = len(y)
        #进行前向传播
        o,s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        # y^t-yt, 用来表示预测值与真实值之间的差
        delta_o = o
        delta_o[np.arange(len(y)),y] -=1.

        # For each output backwards
        for t in np.arange(T)[::-1]:
            # 计算V的梯度
            dLdV += np.outer(delta_o[t], s[t].T)
            # 初始化的delta计算
            delta_t = self.V.T.dot(delta_o[t]) * (1-s[t] ** 2)
            # BPTT
            for bptt_step in np.arange(max(0,t-self.bptt_truncate),t+1)[::-1]:
                # 累加W的梯度
                dLdW += np.outer(delta_t, s[bptt_step-1])
                # 计算U的梯度
                dLdU[:,x[bptt_step]] += delta_t

                #下一步更新delta
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

        return [dLdU, dLdV, dLdW]

    def sgd(self, x, y, learning_rate):
        '''
        随即梯度下降
        Args:
            x: 输入句子序列，一个句子
            y：输出句子序列
            learning_rate: 学习率
        '''
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.W -= learning_rate * dLdW
        self.V -= learning_rate * dLdV
        self.U -= learning_rate * dLdU

    def train_with_sgd(self, train_iter, nepoch = 100, learning_rate = 0.005):
        '''
        用随机梯度下降训练模型
        Args:
            X_train: 训练集
            Y_train: 训练集标签
            nepoch: 轮询次数
            learning_rate: 学习率

        '''
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # optionally evaluate the loss
            for X_train, Y_train in train_iter:
                loss = self.calc_avg_loss(X_train, Y_train)
                losses.append(loss)

                # 对每一个训练样本，进行SGD
                for i in range(len(Y_train)):
                    self.sgd(X_train[i], Y_train[i], learning_rate)
                    num_examples_seen += 1

                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))






