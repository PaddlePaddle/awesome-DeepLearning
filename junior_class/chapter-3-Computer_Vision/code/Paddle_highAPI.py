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
'''
# ## 使用飞桨高层API直接调用网络
# 
# 飞桨框架2.0版本支持全新升级的API体系，除了基础API外，还支持了高层API。通过高低融合实现灵活组网，让飞桨API更简洁、更易用、更强大。高层API支持paddle.vision.models接口，包含了对常用模型的封装，包括ResNet、VGG、MobileNet、LeNet等。使用高层API调用这些网络，可以快速完成神经网络的训练和Fine-tune。
# 
# 代码示例如下：

import paddle
from paddle.vision.models import resnet50

# 调用高层API的resnet50模型
model = resnet50()
# 设置pretrained参数为True，可以加载resnet50在imagenet数据集上的预训练模型
# model = resnet50(pretrained=True)

# 随机生成一个输入
x = paddle.rand([1, 3, 224, 224])
# 得到残差50的计算结果
out = model(x)
# 打印输出的形状，由于resnet50默认的是1000分类
# 所以输出shape是[1x1000]
# print(out.shape)
'''
'''
# 使用paddle.vision中的模型可以简单快速的构建一个深度学习任务，如下示例，仅14行代码即可实现resnet在Cifar10数据集上的训练。
'''

import paddle
# 从paddle.vision.models 模块中import 残差网络，VGG网络，LeNet网络
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.transforms import Transpose

# 确保从paddle.vision.datasets.Cifar10中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')
# 调用resnet50模型
model = paddle.Model(resnet50(pretrained=False, num_classes=10))

# 使用Cifar10数据集
train_dataset = Cifar10(mode='train', transform=Transpose())
val_dataset = Cifar10(mode='test', transform=Transpose())
# 定义优化器
optimizer = Momentum(
    learning_rate=0.01,
    momentum=0.9,
    weight_decay=L2Decay(1e-4),
    parameters=model.parameters())
# 进行训练前准备
model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(train_dataset,
          val_dataset,
          epochs=50,
          batch_size=64,
          save_dir="./output",
          num_workers=8)
