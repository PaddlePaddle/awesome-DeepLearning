import paddle
from paddle.nn import Linear, ReLU
import paddle.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

datafile = "./data/housing.data"
data = np.fromfile(datafile, sep=" ")

# 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
feature_names = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]
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
maximums, minimums, avgs = (
    training_data.max(axis=0),
    training_data.min(axis=0),
    training_data.sum(axis=0) / training_data.shape[0],
)

# 记录数据的归一化参数，在预测时对数据做归一化
max_values = maximums
min_values = minimums
avg_values = avgs

# 对数据进行归一化处理
for i in range(feature_num):
    # print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

# 训练集和测试集的划分比例
training_data = data[:offset]
test_data = data[offset:]
print(training_data.shape, test_data.shape)


class Regressor(paddle.nn.Layer):
    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        self.fc = Linear(in_features=13, out_features=20)
        self.fc2 = Linear(in_features=20, out_features=1)
        self.act = ReLU()

    # 网络的前向计算
    def forward(self, inputs):
        x = self.fc(inputs)
        x = self.act(x)
        x = self.fc2(x)
        return x


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 10  # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [
        training_data[k : k + BATCH_SIZE]
        for k in range(0, len(training_data), BATCH_SIZE)
    ]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获得当前批次训练数据
        x = x.astype(np.float32)
        y = np.array(mini_batch[:, -1:])  # 获得当前批次训练标签（真实房价）
        y = y.astype(np.float32)
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)

        # 前向计算
        predicts = model(house_features)

        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id % 20 == 0:
            print(
                "epoch: {}, iter: {}, loss is: {}".format(
                    epoch_id, iter_id, avg_loss.numpy()
                )
            )

        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()

# 保存模型参数，文件名为LR_model.pdparams
paddle.save(model.state_dict(), "LR_model.pdparams")
print("模型保存成功，模型参数保存在LR_model.pdparams中")

one_data, label = test_data[:, :-1], test_data[:, -1]
one_data = one_data.reshape([test_data.shape[0], -1])
one_data = one_data.astype(np.float32)

model = Regressor()
# 参数为保存模型参数的文件地址
model_dict = paddle.load("LR_model.pdparams")
model.load_dict(model_dict)
model.eval()

# 参数为数据集的文件地址
# 将数据转为动态图的variable格式
one_data = paddle.to_tensor(one_data)
predict = model(one_data)

# 对结果做反归一化处理
predict = predict * (max_values[-1] - min_values[-1]) + avg_values[-1]
# 对label数据做反归一化处理
label = label * (max_values[-1] - min_values[-1]) + avg_values[-1]

print(
    "Inference result is {}, the corresponding label is {}".format(
        predict.numpy(), label
    )
)

t = range(test_data.shape[0])
plt.ylim(0, 70)
plt.plot(t, predict.numpy(), label="predict price")
plt.plot(t, label, label="true price")
plt.show()