import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def main(num_mid):
    # 数据加载
    boston = load_boston()
    x = boston['data']
    y = boston['target']
    # print(x.shape)
    # print(y.shape)

    # 将y转换形状
    y = y.reshape(-1,1)

    # 数据规范化
    ss_input = MinMaxScaler()
    x = ss_input.fit_transform(x)

    # 将数据放到Torch中
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)

    # 数据集切分
    train_x, text_x, train_y, text_y = train_test_split(x, y, test_size=0.25)

    # 构建网络
    model = nn.Sequential(
        nn.Linear(13,num_mid),
        nn.ReLU(),
        nn.Linear(num_mid,1)
    )

    # 定义优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    # 训练
    max_epoch = 500
    iter_loss = []
    for i in range(max_epoch):
        y_pred = model(train_x)
        loss = criterion(y_pred, train_y)
        iter_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # 绘制不同迭代过程的loss
    # x = np.arange(max_epoch)
    # y = np.array(iter_loss)
    # plt.plot(x,y)
    # plt.title('Loss Value in all interations')
    # plt.xlabel('Interation')
    # plt.ylabel('Mean loss Value')
    # plt.show()

    # 测试
    output = model(text_x)
    loss = criterion(output, text_y)
    # predict_list = output.detach().numpy()
    # print(predict_list)
    print("Loss :", loss.item())
    return loss.item()
    # # 真实值与预测值的散点图
    # x = np.arange(text_x.shape[0])
    # line1 = plt.scatter(x,predict_list,c='red',Label='predict')
    # line2 = plt.scatter(x,text_y,c='yellow',Label='real')
    # plt.legend(loc = 'best')
    # plt.title('Prediction Vs Real')
    # plt.ylabel('House Price')
    # plt.show()
    #
    # # 预测与实际标签对比
    # plt.scatter(text_y, predict_list)
    # plt.xlabel("prices")
    # plt.ylabel("predicted prices")
    # plt.show()


if __name__ == '__main__':
    iter_loss = []
    x = []
    for i in range(10, 100, 2):
        loss = main(i)
        x.append(i)
        iter_loss.append(loss)

    plt.plot(x, iter_loss)
    plt.xlabel("Num of neuros")
    plt.ylabel("Loss")
    plt.show()
