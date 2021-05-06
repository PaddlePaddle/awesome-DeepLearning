# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# 导入需要的包
import paddle
from paddle.nn import Conv2D, MaxPool2D, BatchNorm2D, Linear, Dropout


# 定义vgg网络
class VGG(paddle.nn.Layer):
    def __init__(self):
        super(VGG, self).__init__()

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个卷积块，包含两个卷积
        self.conv1_1 = Conv2D(
            in_channels=in_channels[0],
            out_channels=in_channels[1],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv1_2 = Conv2D(
            in_channels=in_channels[1],
            out_channels=in_channels[1],
            kernel_size=3,
            padding=1,
            stride=1)
        # 定义第二个卷积块，包含两个卷积
        self.conv2_1 = Conv2D(
            in_channels=in_channels[1],
            out_channels=in_channels[2],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv2_2 = Conv2D(
            in_channels=in_channels[2],
            out_channels=in_channels[2],
            kernel_size=3,
            padding=1,
            stride=1)
        # 定义第三个卷积块，包含三个卷积
        self.conv3_1 = Conv2D(
            in_channels=in_channels[2],
            out_channels=in_channels[3],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv3_2 = Conv2D(
            in_channels=in_channels[3],
            out_channels=in_channels[3],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv3_3 = Conv2D(
            in_channels=in_channels[3],
            out_channels=in_channels[3],
            kernel_size=3,
            padding=1,
            stride=1)
        # 定义第四个卷积块，包含三个卷积
        self.conv4_1 = Conv2D(
            in_channels=in_channels[3],
            out_channels=in_channels[4],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv4_2 = Conv2D(
            in_channels=in_channels[4],
            out_channels=in_channels[4],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv4_3 = Conv2D(
            in_channels=in_channels[4],
            out_channels=in_channels[4],
            kernel_size=3,
            padding=1,
            stride=1)
        # 定义第五个卷积块，包含三个卷积
        self.conv5_1 = Conv2D(
            in_channels=in_channels[4],
            out_channels=in_channels[5],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv5_2 = Conv2D(
            in_channels=in_channels[5],
            out_channels=in_channels[5],
            kernel_size=3,
            padding=1,
            stride=1)
        self.conv5_3 = Conv2D(
            in_channels=in_channels[5],
            out_channels=in_channels[5],
            kernel_size=3,
            padding=1,
            stride=1)

        # 使用Sequential 将卷积和relu组成一个线性结构（fc + relu）
        # 当输入为224x224时，经过五个卷积块和池化层后，特征维度变为[512x7x7]
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(512 * 7 * 7, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(
            self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential 将卷积和relu组成一个线性结构（fc + relu）
        self.fc2 = paddle.nn.Sequential(
            paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())

        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(
            self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, 1)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
