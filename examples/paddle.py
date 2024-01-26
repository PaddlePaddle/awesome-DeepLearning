#!/usr/bin/env python
# coding: utf-8

# In[4]:


import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np


# In[5]:


def load_data():
    # 从文件导入数据
    datafile = '/home/aistudio/data/data99224/housing.data'
    data = np.fromfile(datafile, sep=' ')
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
        'PTRATIO', 'B', 'LSTAT', 'MEDV'
    ]
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(
        axis=0), training_data.sum(axis=0) / training_data.shape[0]

    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data


# In[6]:


class Regressor(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Regressor, self).__init__(name_scope)
        name_scope = self.full_name()
        self.fc = Linear(input_dim=13, output_dim=1, act=None)

    def forward(self, inputs):
        x = self.fc(inputs)
        return x


with fluid.dygraph.guard():
    model = Regressor("Regressor")
    model.train()
    training_data, test_data = load_data()
    opt = fluid.optimizer.SGD(learning_rate=0.01,
                              parameter_list=model.parameters())

with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10
    BATCH_SIZE = 10

    for epoch_id in range(EPOCH_NUM):
        np.random.shuffle(training_data)
        mini_batches = [
            training_data[k:k + BATCH_SIZE]
            for k in range(0, len(training_data), BATCH_SIZE)
        ]

        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32')
            y = np.array(mini_batch[:, -1:]).astype('float32')
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)

            predicts = model(house_features)

            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id % 20 == 0:
                print("epoch: {}, iter: {}, loss is: {}".format(
                    epoch_id, iter_id, avg_loss.numpy()))

            avg_loss.backward()
            opt.minimize(avg_loss)
            model.clear_gradients()

    fluid.save_dygraph(model.state_dict(), './result/LR_model')


def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    tmp = datas[-10]
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]

    for i in range(len(one_data) - 1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] -
                                                       min_values[i])

    data = np.reshape(np.array(one_data[:-1]), [1, -1]).astype(np.float32)
    label = one_data[-1]
    return data, label


with dygraph.guard():
    model_dict, _ = fluid.load_dygraph('./result/LR_model')
    model.load_dict(model_dict)
    model.eval()

    test_data, label = load_one_example("/home/aistudio/data/data99224/housing.data")
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)

    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(
        results.numpy(), label))


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
