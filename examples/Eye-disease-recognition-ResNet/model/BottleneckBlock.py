import paddle.nn as nn
import paddle

from .ConvBNLayer import ConvBNLayer
# 定义残差块
# 每个残差块会对输入图片做三次卷积，然后跟输入图片进行短接
# 如果残差块中第三次卷积输出特征图的形状和输入不一致，则对输入图片做1x1卷积，将其输出形状调整为一致
class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 shortcut=True,
                 version='O'
                 ):
        super(BottleneckBlock,self).__init__()
        pathA_dict={}
        pathB_dict={}
        # default版本
        pathA_default=nn.Sequential(
            ConvBNLayer(num_channels=num_channels,num_filters=num_filters,filter_size=1,stride=stride,),
            ConvBNLayer(num_channels=num_filters,num_filters=num_filters,filter_size=3,),
            ConvBNLayer(num_channels=num_filters,num_filters=num_filters*4,filter_size=1,act='None'),
        )
        pathB_default=nn.Sequential(
            ConvBNLayer(num_channels=num_channels,num_filters=num_filters*4,filter_size=1,stride=stride,act='None'),
        )
        # B版本修改
        pathA_tweak=nn.Sequential(
            ConvBNLayer(num_channels=num_channels,num_filters=num_filters,filter_size=1,),
            ConvBNLayer(num_channels=num_filters,num_filters=num_filters,filter_size=3,stride=stride,),
            ConvBNLayer(num_channels=num_filters,num_filters=num_filters*4,filter_size=1,),
        )
        pathA_dict['B']=pathA_tweak
        # D 版本修改
        pathB_tweak=nn.Sequential(
            nn.AvgPool2D(kernel_size=stride,stride=stride),
            ConvBNLayer(num_channels=num_channels,num_filters=num_filters*4,filter_size=1),
        )
        pathB_dict['D']=pathB_tweak
        pathA_dict['D']=pathA_tweak
        self.shortcut=shortcut
        self.pathA=pathA_dict.get(version,pathA_default)
        self.pathB=pathB_dict.get(version,pathB_default)
        self._num_channels_out=num_filters*4
    def forward(self,inputs):
        pathA=self.pathA(inputs)
        if self.shortcut:
            pathB=inputs
        else:
            pathB=self.pathB(inputs)
        output=paddle.add(x=pathA,y=pathB)
        output=nn.functional.relu(output)
        return output