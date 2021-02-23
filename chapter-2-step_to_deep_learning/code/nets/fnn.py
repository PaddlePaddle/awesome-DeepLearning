# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

import paddle


# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义一层全连接层，输出维度是1
        self.fc1 = paddle.nn.Linear(in_features=784, out_features=500)
        self.act = paddle.nn.Sigmoid()
        # self.act = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=500, out_features=10)
        self.softmax = paddle.nn.Softmax()

    # 定义网络结构的前向计算过程
    def forward(self, x):
        outputs = self.fc1(x)
        outputs = self.act(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)
        return outputs


# 定义mnist数据识别网络结构，同房价预测网络
class MultiMNIST(paddle.nn.Layer):
    def __init__(self):
        super(MultiMNIST, self).__init__()
        # 定义一层全连接层，输出维度是1
        self.fc1 = paddle.nn.Linear(in_features=784, out_features=512)
        # self.act = paddle.nn.Sigmoid()
        self.act = paddle.nn.ReLU()
        self.fc2 = paddle.nn.Linear(in_features=512, out_features=256)
        self.fc3 = paddle.nn.Linear(in_features=256, out_features=10)
        self.softmax = paddle.nn.Softmax()

    # 定义网络结构的前向计算过程
    def forward(self, x):
        outputs = self.fc1(x)
        outputs = self.act(outputs)
        outputs = self.fc2(outputs)
        outputs = self.act(outputs)
        outputs = self.fc3(outputs)
        outputs = self.softmax(outputs)
        return outputs
