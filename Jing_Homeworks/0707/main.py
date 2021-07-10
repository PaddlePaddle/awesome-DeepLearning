# !usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import paddle
from model import MiniNet
from paddle.optimizer import SGD
from paddle.vision.transforms import Compose, Normalize
import paddle.nn.functional as F

def load_data():
    transform = Compose([Normalize(mean=[127.5],
                                   std=[127.5],
                                   data_format='CHW')])
    print("-----Start Downloading-----")
    train_dataset = paddle.vision.datasets.MNIST(mode='train',transform=transform)
    test_dataset = paddle.vision.datasets.MNIST(mode='test',transform=transform)
    print("-----Downloading Finished-----")
    return train_dataset, test_dataset


def train(model, optimizer, train_dataset, epochs, batch_size):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model.train()
    # 定义数据读取器，训练数据读取器和验证数据读取器
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data[0], data[1]
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 运行模型前向计算，得到预测值
            predicts = model(img)
            loss = F.cross_entropy(predicts, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 200 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch + 1, batch_id, avg_loss.numpy()))
            # 反向传播，更新权重，清除梯度
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()



def test(model, test_dataset, batch_size):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model.eval()
    # 定义数据读取器，训练数据读取器和验证数据读取器
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=batch_size)

    for batch_id, data in enumerate(test_loader()):
        x_data, y_data = data
        img = paddle.to_tensor(x_data)
        label = paddle.to_tensor(y_data)
        # 运行模型前向计算，得到预测值
        predicts = model(img)
        loss = F.cross_entropy(predicts, label)
        avg_loss = paddle.mean(loss)
        acc = paddle.metric.accuracy(predicts, label)
        if batch_id % 200 == 0:
            print("batch_id: {} \t loss: {} \t Acc:{}".format(batch_id, avg_loss.numpy(), acc.numpy()))



def main():
    train_dataset, test_dataset = load_data()
    model = MiniNet()
    optimizer = SGD(learning_rate=3e-2, parameters=model.parameters())
    epochs = 10
    batch_size = 128
    testing_batch_size = 1000

    train(model, optimizer, train_dataset, epochs, batch_size)
    test(model, test_dataset, testing_batch_size)


if __name__ == '__main__':
    main()

