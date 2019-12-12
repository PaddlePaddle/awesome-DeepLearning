# -*- coding: utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The implementation of neural network using Paddle.
"""
# -*- coding: utf-8 -*-
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import FC
import numpy as np


def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
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

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                               training_data.sum(axis=0) / training_data.shape[0]

    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    # ratio = 0.8
    # offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


class Regressor(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Regressor, self).__init__(name_scope)
        name_scope = self.full_name()
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = FC(name_scope, size=1, act=None)

    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x


def train():
    # 定义飞桨动态图的工作环境
    with fluid.dygraph.guard():
        # 声明定义好的线性回归模型
        model = Regressor("Regressor")
        # 开启模型训练模式
        model.train()
        # 加载数据
        training_data, test_data = load_data()
        # 定义优化算法，这里使用随机梯度下降-SGD
        # 学习率设置为0.01
        opt = fluid.optimizer.SGD(learning_rate=0.01)

        EPOCH_NUM = 10  # 设置外层循环次数
        BATCH_SIZE = 10  # 设置batch大小

        # 定义外层循环
        for epoch_id in range(EPOCH_NUM):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个batch包含10条数据
            mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
            # 定义内层循环
            for iter_id, mini_batch in enumerate(mini_batches):
                x = np.array(mini_batch[:, :-1]).astype('float32')  # 获得当前批次训练数据
                y = np.array(mini_batch[:, -1:]).astype('float32')  # 获得当前批次训练标签（真实房价）
                # 将numpy数据转为飞桨动态图variable形式
                house_features = dygraph.to_variable(x)
                prices = dygraph.to_variable(y)

                # 前向计算
                predicts = model(house_features)

                # 计算损失
                loss = fluid.layers.square_error_cost(predicts, label=prices)
                avg_loss = fluid.layers.mean(fluid.layers.sqrt(loss))
                if iter_id % 20 == 0:
                    print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))

                # 反向传播
                avg_loss.backward()
                # 最小化loss,更新参数
                opt.minimize(avg_loss)
                # 清除梯度
                model.clear_gradients()
        # 保存模型
        fluid.save_dygraph(model.state_dict(), 'LR_model')


def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # 选择倒数第10条数据用于测试
    tmp = datas[-10]
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]

    # 对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] - min_values[i])

    data = np.reshape(np.array(one_data[:-1]), [1, -1]).astype(np.float32)
    label = one_data[-1]
    return data, label


def valid():
    # 开启动态图工作环境
    with dygraph.guard():
        # 声明定义好的线性回归模型
        model = Regressor("Regressor")
        # 开启模型训练模式
        model.eval()
        # 参数为保存模型参数的文件地址
        model_dict, _ = fluid.load_dygraph('LR_model')
        model.load_dict(model_dict)
        model.eval()

        # 参数为数据集的文件地址
        test_data, label = load_one_example('./work/housing.data')
        # 将数据转为动态图的variable格式
        test_data = dygraph.to_variable(test_data)
        results = model(test_data)

        # 对结果做反归一化处理
        results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
        print("Inference result is {}, the corresponding label is {}".format(results.numpy(), label))


if __name__ == '__main__':
    train()
    valid()