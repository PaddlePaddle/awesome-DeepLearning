# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
'''
# LeNet在手写数字识别上的应用
#
# LeNet网络的实现代码如下：
'''

# 导入需要的包
import paddle
import numpy as np
import paddle.nn.functional as F
from nets.LeNet import LeNet


# 定义训练过程
def train(model):
    # 开启0号GPU训练
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    print('start training ... ')
    model.train()
    epoch_num = 5
    opt = paddle.optimizer.Momentum(
        learning_rate=0.001, momentum=0.9, parameters=model.parameters())

    # 使用Paddle自带的数据读取器
    train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=10)
    valid_loader = paddle.batch(paddle.dataset.mnist.test(), batch_size=10)
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            # 调整输入数据形状和类型
            x_data = np.array(
                [item[0] for item in data], dtype='float32').reshape(-1, 1, 28,
                                                                     28)
            y_data = np.array(
                [item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 计算模型输出
            logits = model(img)
            # 计算损失函数
            loss = F.softmax_with_cross_entropy(logits, label)
            avg_loss = paddle.mean(loss)

            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(
                    epoch, batch_id, avg_loss.numpy()))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            # 调整输入数据形状和类型
            x_data = np.array(
                [item[0] for item in data], dtype='float32').reshape(-1, 1, 28,
                                                                     28)
            y_data = np.array(
                [item[1] for item in data], dtype='int64').reshape(-1, 1)
            # 将numpy.ndarray转化成Tensor
            img = paddle.to_tensor(x_data)
            label = paddle.to_tensor(y_data)
            # 计算模型输出
            logits = model(img)
            pred = F.softmax(logits)
            # 计算损失函数
            loss = F.softmax_with_cross_entropy(logits, label)
            acc = paddle.metric.accuracy(pred, label)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())
        print("[validation] accuracy/loss: {}/{}".format(
            np.mean(accuracies), np.mean(losses)))
        model.train()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')


# 创建模型
model = LeNet(num_classes=10)
# 启动训练过程
train(model)
