import sys
import torch
import torch.nn as nn
import  matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(2,15),
            nn.ReLU(),
            nn.Linear(15,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        classfication = self.classify(x)
        return classfication


def define_cnn_model():
    model = Sequential()
    # 卷积层
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=(200, 200, 3)))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))
    # Flatten 层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128,
                    activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
