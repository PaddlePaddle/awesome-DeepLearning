import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


# NNLM Parameter
n_step = 5 # 窗口大小
m = 40 # 词嵌入向量大小
n_hidden = 20 # 隐层神经元个数


# Model
class NNLM(nn.Module):
    def __init__(self,n_class):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)
        self.fc1 = nn.Linear(n_step*m,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_class)
        self.fc3 = nn.Linear(n_step*m,n_class)

    def forward(self, X):
        X = self.C(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * m]
        output = torch.tanh(self.fc1(X))
        output = self.fc2(output) + self.fc3(X)

        return output


