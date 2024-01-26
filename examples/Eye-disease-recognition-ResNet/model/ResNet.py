import paddle.nn as nn
import paddle

from .ConvBNLayer import ConvBNLayer
from .BottleneckBlock import BottleneckBlock


# 定义ResNet模型
class ResNet(nn.Layer):
    def __init__(self, layers=50, class_dim=10, version='O'):
        """
        layers,网络层数，可以可选项：50,101,152
        class_dim,分类标签的类别数
        """
        super(ResNet, self).__init__()
        self.version = version
        self.layers = layers
        self.max_accuracy = 0.0

        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        # ResNet50包含的stage1-4模块分别包括3,4,6,3个残差块
        if layers == 50:
            depth = [3, 4, 6, 3]
        # ResNet101包含的stage1-4模块分别包括3,4,23,3个残差块
        if layers == 101:
            depth = [3, 4, 23, 3]
        # ResNet152包含的stage1-4分别包括3,8,36,3个残差块
        if layers == 152:
            depth = [3, 8, 36, 3]
        # stage1-4所使用残差块的输出通道数
        num_filters = [64, 128, 256, 512]

        # input stem模块,default版本：64个7x7的卷积加上一个3x3最大化池化层，步长均为2
        input_stem_dict = {}
        input_stem_default = nn.Sequential(
            ConvBNLayer(num_channels=3, num_filters=64, filter_size=7, stride=2, ),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ),
        )
        # C版本修改
        input_stem_tweak = nn.Sequential(
            ConvBNLayer(num_channels=3, num_filters=64, filter_size=3, stride=2, ),
            ConvBNLayer(num_channels=64, num_filters=64, filter_size=3, ),
            ConvBNLayer(num_channels=64, num_filters=64, filter_size=3, ),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1, ),
        )
        input_stem_dict['C'] = input_stem_tweak

        self.input_stem = input_stem_dict.get(version, input_stem_default)

        # stage1-4模块，使用各个残差块进行卷积操作
        self.bottleneck_block_list = []
        num_channels = 64
        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        version=version))
                num_channels = bottleneck_block._num_channels_out
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True

        # 在stage4的输出特征图上使用全局池化
        self.pool2d_avg = nn.AdaptiveAvgPool2D(output_size=1)

        # stdv用来作为全连接层随机初始化参数的方差
        import math
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        # 创建全连接层，输出大小为类别数目，经过残差网络的卷积核全局池化后，
        # 卷积特征的维度是[B,2048,1,1]，故最后一层全连接层的输入维度是2048
        self.out = nn.Linear(in_features=2048, out_features=class_dim,
                             weight_attr=paddle.ParamAttr(
                                 initializer=paddle.nn.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        x = self.input_stem(inputs)
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        x = self.pool2d_avg(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.out(x)
        return x