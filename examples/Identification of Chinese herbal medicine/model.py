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
# coding=utf-8
import paddle


# 定义卷积池化网络
class ConvPool(paddle.nn.Layer):
    '''卷积+池化'''

    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            pool_size,
            pool_stride,
            groups,
            conv_stride=1,
            conv_padding=1, ):
        super(ConvPool, self).__init__()

        # groups代表卷积层的数量
        for i in range(groups):
            self.add_sublayer(  #添加子层实例
                'bb_%d' % i,
                paddle.nn.Conv2D(  # layer
                    in_channels=num_channels,  #通道数
                    out_channels=num_filters,  #卷积核个数
                    kernel_size=filter_size,  #卷积核大小
                    stride=conv_stride,  #步长
                    padding=conv_padding,  #padding
                ))
            self.add_sublayer('relu%d' % i, paddle.nn.ReLU())
            num_channels = num_filters

        self.add_sublayer(
            'Maxpool',
            paddle.nn.MaxPool2D(
                kernel_size=pool_size,  #池化核大小
                stride=pool_stride  #池化步长
            ))

    def forward(self, inputs):
        x = inputs
        for prefix, sub_layer in self.named_children():
            # print(prefix,sub_layer)
            x = sub_layer(x)
        return x


# VGG网络
class VGGNet(paddle.nn.Layer):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        # 5个卷积池化操作
        self.convpool01 = ConvPool(
            3, 64, 3, 2, 2, 2)  #3:通道数，64：卷积核个数，3:卷积核大小，2:池化核大小，2:池化步长，2:连续卷积个数
        self.convpool02 = ConvPool(64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(128, 256, 3, 2, 2, 3)
        self.convpool04 = ConvPool(256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(512, 512, 3, 2, 2, 3)
        self.pool_5_shape = 512 * 7 * 7
        # 三个全连接层
        self.fc01 = paddle.nn.Linear(self.pool_5_shape, 4096)
        self.drop1 = paddle.nn.Dropout(p=0.5)
        self.fc02 = paddle.nn.Linear(4096, 4096)
        self.drop2 = paddle.nn.Dropout(p=0.5)
        self.fc03 = paddle.nn.Linear(4096, num_classes)

    def forward(self, inputs, label=None):
        # print('input_shape:', inputs.shape) #[8, 3, 224, 224]
        """前向计算"""
        out = self.convpool01(inputs)
        # print('convpool01_shape:', out.shape)           #[8, 64, 112, 112]
        out = self.convpool02(out)
        # print('convpool02_shape:', out.shape)           #[8, 128, 56, 56]
        out = self.convpool03(out)
        # print('convpool03_shape:', out.shape)           #[8, 256, 28, 28]
        out = self.convpool04(out)
        # print('convpool04_shape:', out.shape)           #[8, 512, 14, 14]
        out = self.convpool05(out)
        # print('convpool05_shape:', out.shape)           #[8, 512, 7, 7]

        out = paddle.reshape(out, shape=[-1, 512 * 7 * 7])
        out = self.fc01(out)
        out = self.drop1(out)
        out = self.fc02(out)
        out = self.drop2(out)
        out = self.fc03(out)

        if label is not None:
            acc = paddle.metric.accuracy(input=out, label=label)
            return out, acc
        else:
            return out
