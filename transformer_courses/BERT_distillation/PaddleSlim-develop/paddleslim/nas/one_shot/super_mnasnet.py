import paddle
from paddle import fluid
from paddle.fluid.layer_helper import LayerHelper
import numpy as np
from one_shot_nas import OneShotSuperNet

__all__ = ['SuperMnasnet']


class DConvBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 in_channels,
                 channels,
                 expansion,
                 stride,
                 kernel_size=3,
                 padding=1):
        super(DConvBlock, self).__init__(name_scope)
        self.expansion = expansion
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.flops = 0
        self.flops_calculated = False
        self.expand = fluid.dygraph.Conv2D(
            in_channels,
            num_filters=in_channels * expansion,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            bias_attr=False)
        self.expand_bn = fluid.dygraph.BatchNorm(
            num_channels=in_channels * expansion, act='relu6')

        self.dconv = fluid.dygraph.Conv2D(
            in_channels * expansion,
            num_filters=in_channels * expansion,
            filter_size=kernel_size,
            stride=stride,
            padding=padding,
            act=None,
            bias_attr=False,
            groups=in_channels * expansion,
            use_cudnn=False)
        self.dconv_bn = fluid.dygraph.BatchNorm(
            num_channels=in_channels * expansion, act='relu6')

        self.project = fluid.dygraph.Conv2D(
            in_channels * expansion,
            num_filters=channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            bias_attr=False)
        self.project_bn = fluid.dygraph.BatchNorm(
            num_channels=channels, act=None)

        self.shortcut = fluid.dygraph.Conv2D(
            in_channels,
            num_filters=channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            bias_attr=False)
        self.shortcut_bn = fluid.dygraph.BatchNorm(
            num_channels=channels, act=None)

    def get_flops(self, input, output, op):
        if not self.flops_calculated:
            flops = input.shape[1] * output.shape[1] * (
                op._filter_size**2) * output.shape[2] * output.shape[3]
            if op._groups:
                flops /= op._groups
            self.flops += flops

    def forward(self, inputs):
        expand_x = self.expand_bn(self.expand(inputs))
        self.get_flops(inputs, expand_x, self.expand)
        dconv_x = self.dconv_bn(self.dconv(expand_x))
        self.get_flops(expand_x, dconv_x, self.dconv)
        proj_x = self.project_bn(self.project(dconv_x))
        self.get_flops(dconv_x, proj_x, self.project)
        if self.in_channels != self.channels and self.stride == 1:
            shortcut = self.shortcut_bn(self.shortcut(inputs))
            self.get_flops(inputs, shortcut, self.shortcut)
        elif self.stride == 1:
            shortcut = inputs
        self.flops_calculated = True
        if self.stride == 1:
            out = fluid.layers.elementwise_add(x=proj_x, y=shortcut)
            return out
        return proj_x


class SearchBlock(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 in_channels,
                 channels,
                 stride,
                 kernel_size=3,
                 padding=1):
        super(SearchBlock, self).__init__(name_scope)
        self._stride = stride
        self.block_list = []
        self.flops = [0 for i in range(10)]
        self.flops_calculated = [False if i < 6 else True for i in range(10)]
        kernels = [3, 5, 7]
        expansions = [3, 6]
        for k in kernels:
            for e in expansions:
                self.block_list.append(
                    DConvBlock(self.full_name(), in_channels, channels, e,
                               stride, k, (k - 1) // 2))
                self.add_sublayer("expansion_{}_kernel_{}".format(e, k),
                                  self.block_list[-1])

    def forward(self, inputs, arch):
        if arch >= 6:
            return inputs
        out = self.block_list[arch](inputs)
        if not self.flops_calculated[arch]:
            self.flops[arch] = self.block_list[arch].flops
            self.flops_calculated[arch] = True
        return out


class AuxiliaryHead(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes):
        super(AuxiliaryHead, self).__init__(name_scope)

        self.pool1 = fluid.dygraph.Pool2D(
            5, 'avg', pool_stride=3, pool_padding=0)
        self.conv1 = fluid.dygraph.Conv2D(128, 1, bias_attr=False)
        self.bn1 = fluid.dygraph.BatchNorm(128, act='relu6')
        self.conv2 = fluid.dygraph.Conv2D(768, 2, bias_attr=False)
        self.bn2 = fluid.dygraph.BatchNorm(768, act='relu6')
        self.classifier = fluid.dygraph.FC(num_classes, act='softmax')
        self.layer_helper = LayerHelper(self.full_name(), act='relu6')

    def forward(self, inputs):  #pylint: disable=arguments-differ
        inputs = self.layer_helper.append_activation(inputs)
        inputs = self.pool1(inputs)
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.bn2(inputs)
        inputs = self.classifier(inputs)
        return inputs


class SuperMnasnet(OneShotSuperNet):
    def __init__(self,
                 name_scope,
                 input_channels=3,
                 out_channels=1280,
                 repeat_times=[6, 6, 6, 6, 6, 6],
                 stride=[1, 1, 1, 1, 2, 1],
                 channels=[16, 24, 40, 80, 96, 192, 320],
                 use_auxhead=False):
        super(SuperMnasnet, self).__init__(name_scope)
        self.flops = 0
        self.repeat_times = repeat_times
        self.flops_calculated = False
        self.last_tokens = None
        self._conv = fluid.dygraph.Conv2D(
            input_channels, 32, 3, 1, 1, act=None, bias_attr=False)
        self._bn = fluid.dygraph.BatchNorm(32, act='relu6')
        self._sep_conv = fluid.dygraph.Conv2D(
            32,
            32,
            3,
            1,
            1,
            groups=32,
            act=None,
            use_cudnn=False,
            bias_attr=False)
        self._sep_conv_bn = fluid.dygraph.BatchNorm(32, act='relu6')
        self._sep_project = fluid.dygraph.Conv2D(
            32, 16, 1, 1, 0, act=None, bias_attr=False)
        self._sep_project_bn = fluid.dygraph.BatchNorm(16, act='relu6')

        self._final_conv = fluid.dygraph.Conv2D(
            320, out_channels, 1, 1, 0, act=None, bias_attr=False)
        self._final_bn = fluid.dygraph.BatchNorm(out_channels, act='relu6')
        self.stride = stride
        self.block_list = []
        self.use_auxhead = use_auxhead

        for _iter, _stride in enumerate(self.stride):
            repeat_block = []
            for _ind in range(self.repeat_times[_iter]):
                if _ind == 0:
                    block = SearchBlock(self.full_name(), channels[_iter],
                                        channels[_iter + 1], _stride)
                else:
                    block = SearchBlock(self.full_name(), channels[_iter + 1],
                                        channels[_iter + 1], 1)
                self.add_sublayer("block_{}_{}".format(_iter, _ind), block)
                repeat_block.append(block)
            self.block_list.append(repeat_block)
        if self.use_auxhead:
            self.auxhead = AuxiliaryHead(self.full_name(), 10)

    def init_tokens(self):
        return [
            3, 3, 6, 6, 6, 6, 3, 3, 3, 6, 6, 6, 3, 3, 3, 3, 6, 6, 3, 3, 3, 6,
            6, 6, 3, 3, 3, 6, 6, 6, 3, 6, 6, 6, 6, 6
        ]

    def range_table(self):
        max_v = [
            6, 6, 10, 10, 10, 10, 6, 6, 6, 10, 10, 10, 6, 6, 6, 6, 10, 10, 6,
            6, 6, 10, 10, 10, 6, 6, 6, 10, 10, 10, 6, 10, 10, 10, 10, 10
        ]
        return (len(max_v) * [0], max_v)

    def get_flops(self, input, output, op):
        if not self.flops_calculated:
            flops = input.shape[1] * output.shape[1] * (
                op._filter_size**2) * output.shape[2] * output.shape[3]
            if op._groups:
                flops /= op._groups
            self.flops += flops

    def _forward_impl(self, inputs, tokens=None):
        if isinstance(tokens, np.ndarray) and not (tokens == self.last_tokens).all()\
           or not isinstance(tokens, np.ndarray) and not tokens == self.last_tokens:
            self.flops_calculated = False
            self.flops = 0
        self.last_tokens = tokens
        x = self._bn(self._conv(inputs))
        self.get_flops(inputs, x, self._conv)
        sep_x = self._sep_conv_bn(self._sep_conv(x))
        self.get_flops(x, sep_x, self._sep_conv)
        proj_x = self._sep_project_bn(self._sep_project(sep_x))
        self.get_flops(sep_x, proj_x, self._sep_project)
        x = proj_x
        for ind in range(len(self.block_list)):
            for b_ind, block in enumerate(self.block_list[ind]):
                x = fluid.layers.dropout(block(x, tokens[ind * 6 + b_ind]), 0.)
                if not self.flops_calculated:
                    self.flops += block.flops[tokens[ind * 6 + b_ind]]
            if ind == len(self.block_list) * 2 // 3 - 1 and self.use_auxhead:
                fc_aux = self.auxhead(x)
        final_x = self._final_bn(self._final_conv(x))
        self.get_flops(x, final_x, self._final_conv)
        #        x = self.global_pooling(final_x)
        self.flops_calculated = True
        if self.use_auxhead:
            return final_x, fc_aux
        return final_x
