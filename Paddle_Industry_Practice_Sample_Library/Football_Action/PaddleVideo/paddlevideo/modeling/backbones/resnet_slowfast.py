# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal
from ..registry import BACKBONES
from paddlevideo.utils.multigrid import get_norm
import sys
import numpy as np
import paddle.distributed as dist

# seed random seed
paddle.framework.seed(0)


# get init parameters for conv layer
def get_conv_init(fan_out):
    return KaimingNormal(fan_in=fan_out)


def get_bn_param_attr(bn_weight=1.0, coeff=0.0):
    param_attr = paddle.ParamAttr(
        initializer=paddle.nn.initializer.Constant(bn_weight),
        regularizer=paddle.regularizer.L2Decay(coeff))
    return param_attr


"""Video models."""


class BottleneckTransform(paddle.nn.Layer):
    """
    Bottleneck transformation: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 dim_inner,
                 num_groups,
                 stride_1x1=False,
                 inplace_relu=True,
                 eps=1e-5,
                 dilation=1,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(BottleneckTransform, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._eps = eps
        self._stride_1x1 = stride_1x1
        self.norm_module = norm_module
        self._construct(dim_in, dim_out, stride, dim_inner, num_groups,
                        dilation)

    def _construct(self, dim_in, dim_out, stride, dim_inner, num_groups,
                   dilation):
        str1x1, str3x3 = (stride, 1) if self._stride_1x1 else (1, stride)

        fan = (dim_inner) * (self.temp_kernel_size * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self.a = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.a_bn = self.norm_module(num_features=dim_inner,
                                     epsilon=self._eps,
                                     weight_attr=get_bn_param_attr(),
                                     bias_attr=get_bn_param_attr(bn_weight=0.0))

        # 1x3x3, BN, ReLU.
        fan = (dim_inner) * (1 * 3 * 3)
        initializer_tmp = get_conv_init(fan)

        self.b = paddle.nn.Conv3D(
            in_channels=dim_inner,
            out_channels=dim_inner,
            kernel_size=[1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            dilation=[1, dilation, dilation],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.b_bn = self.norm_module(num_features=dim_inner,
                                     epsilon=self._eps,
                                     weight_attr=get_bn_param_attr(),
                                     bias_attr=get_bn_param_attr(bn_weight=0.0))

        # 1x1x1, BN.
        fan = (dim_out) * (1 * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self.c = paddle.nn.Conv3D(
            in_channels=dim_inner,
            out_channels=dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self.c_bn = self.norm_module(
            num_features=dim_out,
            epsilon=self._eps,
            weight_attr=get_bn_param_attr(bn_weight=0.0),
            bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = F.relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = F.relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(paddle.nn.Layer):
    """
    Residual block.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 temp_kernel_size,
                 stride,
                 dim_inner,
                 num_groups=1,
                 stride_1x1=False,
                 inplace_relu=True,
                 eps=1e-5,
                 dilation=1,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            trans_func (string): transform function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            dilation (int): size of dilation.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._eps = eps
        self.norm_module = norm_module
        self._construct(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            fan = (dim_out) * (1 * 1 * 1)
            initializer_tmp = get_conv_init(fan)
            self.branch1 = paddle.nn.Conv3D(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
                bias_attr=False,
                dilation=1)
            self.branch1_bn = self.norm_module(
                num_features=dim_out,
                epsilon=self._eps,
                weight_attr=get_bn_param_attr(),
                bias_attr=get_bn_param_attr(bn_weight=0.0))

        self.branch2 = BottleneckTransform(dim_in,
                                           dim_out,
                                           temp_kernel_size,
                                           stride,
                                           dim_inner,
                                           num_groups,
                                           stride_1x1=stride_1x1,
                                           inplace_relu=inplace_relu,
                                           dilation=dilation,
                                           norm_module=self.norm_module)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x1 = self.branch1(x)
            x1 = self.branch1_bn(x1)
            x2 = self.branch2(x)
            x = paddle.add(x=x1, y=x2)
        else:
            x2 = self.branch2(x)
            x = paddle.add(x=x, y=x2)

        x = F.relu(x)
        return x


class ResStage(paddle.nn.Layer):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        multi-pathway (SlowFast) cases.  More details can be found here:

        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 stride,
                 temp_kernel_sizes,
                 num_blocks,
                 dim_inner,
                 num_groups,
                 num_block_temp_kernel,
                 dilation,
                 stride_1x1=False,
                 inplace_relu=True,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different pathway.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different pathway.
            num_blocks (list): list of p numbers of blocks for each of the
                pathway.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different pathways.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            dilation (list): size of dilation for each pathway.
        """
        super(ResStage, self).__init__()
        assert all((num_block_temp_kernel[i] <= num_blocks[i]
                    for i in range(len(temp_kernel_sizes))))
        self.num_blocks = num_blocks
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[:num_block_temp_kernel[i]] +
            [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (len({
            len(dim_in),
            len(dim_out),
            len(temp_kernel_sizes),
            len(stride),
            len(num_blocks),
            len(dim_inner),
            len(num_groups),
            len(num_block_temp_kernel),
        }) == 1)
        self.num_pathways = len(self.num_blocks)
        self.norm_module = norm_module
        self._construct(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _construct(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):

        for pathway in range(self.num_pathways):
            for i in range(self.num_blocks[pathway]):
                res_block = ResBlock(
                    dim_in[pathway] if i == 0 else dim_out[pathway],
                    dim_out[pathway],
                    self.temp_kernel_sizes[pathway][i],
                    stride[pathway] if i == 0 else 1,
                    dim_inner[pathway],
                    num_groups[pathway],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[pathway],
                    norm_module=self.norm_module)
                self.add_sublayer("pathway{}_res{}".format(pathway, i),
                                  res_block)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]

            for i in range(self.num_blocks[pathway]):
                m = getattr(self, "pathway{}_res{}".format(pathway, i))
                x = m(x)
            output.append(x)

        return output


class ResNetBasicStem(paddle.nn.Layer):
    """
    ResNe(X)t 3D stem module.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        super(ResNetBasicStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        fan = (dim_out) * (self.kernel[0] * self.kernel[1] * self.kernel[2])
        initializer_tmp = get_conv_init(fan)

        self._conv = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = self.norm_module(num_features=dim_out,
                                    epsilon=self.eps,
                                    weight_attr=get_bn_param_attr(),
                                    bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x = self._conv(x)
        x = self._bn(x)
        x = F.relu(x)

        x = F.max_pool3d(x=x,
                         kernel_size=[1, 3, 3],
                         stride=[1, 2, 2],
                         padding=[0, 1, 1],
                         data_format="NCDHW")
        return x


class VideoModelStem(paddle.nn.Layer):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for slow and fast pathways.
    """
    def __init__(self,
                 dim_in,
                 dim_out,
                 kernel,
                 stride,
                 padding,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            eps (float): epsilon for batch norm.
        """
        super(VideoModelStem, self).__init__()

        assert (len({
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
            len(padding),
        }) == 1), "Input pathway dimensions are not consistent."
        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.eps = eps
        self.norm_module = norm_module
        self._construct_stem(dim_in, dim_out)

    def _construct_stem(self, dim_in, dim_out):
        for pathway in range(len(dim_in)):
            stem = ResNetBasicStem(dim_in[pathway], dim_out[pathway],
                                   self.kernel[pathway], self.stride[pathway],
                                   self.padding[pathway], self.eps,
                                   self.norm_module)
            self.add_sublayer("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (len(x) == self.num_pathways
                ), "Input tensor does not contain {} pathway".format(
                    self.num_pathways)

        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])

        return x


class FuseFastToSlow(paddle.nn.Layer):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """
    def __init__(self,
                 dim_in,
                 fusion_conv_channel_ratio,
                 fusion_kernel,
                 alpha,
                 fuse_bn_relu=1,
                 eps=1e-5,
                 norm_module=paddle.nn.BatchNorm3D):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
        """
        super(FuseFastToSlow, self).__init__()
        self.fuse_bn_relu = fuse_bn_relu
        fan = (dim_in * fusion_conv_channel_ratio) * (fusion_kernel * 1 * 1)
        initializer_tmp = get_conv_init(fan)

        self._conv_f2s = paddle.nn.Conv3D(
            in_channels=dim_in,
            out_channels=dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            weight_attr=paddle.ParamAttr(initializer=initializer_tmp),
            bias_attr=False)
        self._bn = norm_module(num_features=dim_in * fusion_conv_channel_ratio,
                               epsilon=eps,
                               weight_attr=get_bn_param_attr(),
                               bias_attr=get_bn_param_attr(bn_weight=0.0))

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self._conv_f2s(x_f)
        #  TODO: For AVA, set fuse_bn_relu=1, check mAP's improve.
        if self.fuse_bn_relu:
            fuse = self._bn(fuse)
            fuse = F.relu(fuse)
        x_s_fuse = paddle.concat(x=[x_s, fuse], axis=1, name=None)

        return [x_s_fuse, x_f]


@BACKBONES.register()
class ResNetSlowFast(paddle.nn.Layer):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """
    def __init__(
        self,
        alpha,
        beta,
        bn_norm_type="batchnorm",
        bn_num_splits=1,
        num_pathways=2,
        depth=50,
        num_groups=1,
        input_channel_num=[3, 3],
        width_per_group=64,
        fusion_conv_channel_ratio=2,
        fusion_kernel_sz=7,  #5?
        pool_size_ratio=[[1, 1, 1], [1, 1, 1]],
        fuse_bn_relu = 1,
        spatial_strides = [[1, 1], [2, 2], [2, 2], [2, 2]],
        use_pool_af_s2 = 1,
    ):
        """
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNetSlowFast, self).__init__()

        self.alpha = alpha  #8
        self.beta = beta  #8
        self.norm_module = get_norm(bn_norm_type, bn_num_splits)
        self.num_pathways = num_pathways
        self.depth = depth
        self.num_groups = num_groups
        self.input_channel_num = input_channel_num
        self.width_per_group = width_per_group
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_sz = fusion_kernel_sz  # NOTE: modify to 7 in 8*8, 5 in old implement
        self.pool_size_ratio = pool_size_ratio
        self.fuse_bn_relu = fuse_bn_relu
        self.spatial_strides = spatial_strides
        self.use_pool_af_s2 = use_pool_af_s2
        self._construct_network()

    def _construct_network(self):
        """
        Builds a SlowFast model.
        The first pathway is the Slow pathway
        and the second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        temp_kernel = [
            [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
            [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
            [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
            [[3], [3]],
        ]  # res5 temporal kernel for slow and fast pathway.

        self.s1 = VideoModelStem(
            dim_in=self.input_channel_num,
            dim_out=[self.width_per_group, self.width_per_group // self.beta],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module)
        self.s1_fuse = FuseFastToSlow(
            dim_in=self.width_per_group // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
            fuse_bn_relu=self.fuse_bn_relu)

        # ResNet backbone
        MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3)}
        (d2, d3, d4, d5) = MODEL_STAGE_DEPTH[self.depth]

        num_block_temp_kernel = [[3, 3], [4, 4], [6, 6], [3, 3]]
        spatial_dilations = [[1, 1], [1, 1], [1, 1], [1, 1]]
        spatial_strides = self.spatial_strides
        #spatial_strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
        #spatial_strides = [[1, 1], [2, 2], [2, 2], [1, 1]] #TODO:check which value is FAIR's impliment

        out_dim_ratio = self.beta // self.fusion_conv_channel_ratio  #4
        dim_inner = self.width_per_group * self.num_groups  #64

        self.s2 = ResStage(dim_in=[
            self.width_per_group + self.width_per_group // out_dim_ratio,
            self.width_per_group // self.beta,
        ],
                           dim_out=[
                               self.width_per_group * 4,
                               self.width_per_group * 4 // self.beta,
                           ],
                           dim_inner=[dim_inner, dim_inner // self.beta],
                           temp_kernel_sizes=temp_kernel[1],
                           stride=spatial_strides[0],
                           num_blocks=[d2] * 2,
                           num_groups=[self.num_groups] * 2,
                           num_block_temp_kernel=num_block_temp_kernel[0],
                           dilation=spatial_dilations[0],
                           norm_module=self.norm_module)

        self.s2_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 4 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
            fuse_bn_relu=self.fuse_bn_relu,
        )

        self.s3 = ResStage(
            dim_in=[
                self.width_per_group * 4 +
                self.width_per_group * 4 // out_dim_ratio,
                self.width_per_group * 4 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 8,
                self.width_per_group * 8 // self.beta,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // self.beta],
            temp_kernel_sizes=temp_kernel[2],
            stride=spatial_strides[1],
            num_blocks=[d3] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[1],
            dilation=spatial_dilations[1],
            norm_module=self.norm_module,
        )

        self.s3_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 8 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
            fuse_bn_relu=self.fuse_bn_relu,
        )

        self.s4 = ResStage(
            dim_in=[
                self.width_per_group * 8 +
                self.width_per_group * 8 // out_dim_ratio,
                self.width_per_group * 8 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 16,
                self.width_per_group * 16 // self.beta,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // self.beta],
            temp_kernel_sizes=temp_kernel[3],
            stride=spatial_strides[2],
            num_blocks=[d4] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[2],
            dilation=spatial_dilations[2],
            norm_module=self.norm_module,
        )

        self.s4_fuse = FuseFastToSlow(
            dim_in=self.width_per_group * 16 // self.beta,
            fusion_conv_channel_ratio=self.fusion_conv_channel_ratio,
            fusion_kernel=self.fusion_kernel_sz,
            alpha=self.alpha,
            norm_module=self.norm_module,
            fuse_bn_relu=self.fuse_bn_relu,
        )

        self.s5 = ResStage(
            dim_in=[
                self.width_per_group * 16 +
                self.width_per_group * 16 // out_dim_ratio,
                self.width_per_group * 16 // self.beta,
            ],
            dim_out=[
                self.width_per_group * 32,
                self.width_per_group * 32 // self.beta,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // self.beta],
            temp_kernel_sizes=temp_kernel[4],
            stride=spatial_strides[3],
            num_blocks=[d5] * 2,
            num_groups=[self.num_groups] * 2,
            num_block_temp_kernel=num_block_temp_kernel[3],
            dilation=spatial_dilations[3],
            norm_module=self.norm_module,
        )

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.s1(x)  #VideoModelStem
        x = self.s1_fuse(x)  #FuseFastToSlow
        x = self.s2(x)  #ResStage
        x = self.s2_fuse(x)

        #  TODO: For AVA, set use_pool_af_s2=1, check mAP's improve.
        if self.use_pool_af_s2:
            for pathway in range(self.num_pathways):
                x[pathway] = F.max_pool3d(x=x[pathway],
                                          kernel_size=self.pool_size_ratio[pathway],
                                          stride=self.pool_size_ratio[pathway],
                                          padding=[0, 0, 0],
                                          data_format="NCDHW")

        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        return x
