# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import math
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import BatchNorm2D, Conv2D
from paddle.nn.initializer import Constant, Normal
from paddle.vision.models import ResNet

from ...utils import load_ckpt
from ..registry import BACKBONES
from ..weight_init import kaiming_normal_, _calculate_fan_in_and_fan_out

zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
normal_ = Normal(mean=0, std=1e-3)


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def gram_matrix(y):
    (b, ch, h, w) = y.shape
    features = y.reshape([b, ch, w * h])
    features_t = paddle.transpose(features, [0, 2, 1])
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def convt_bn_relu(in_channels,
                  out_channels,
                  kernel_size,
                  stride=1,
                  padding=0,
                  output_padding=0,
                  bn=True,
                  relu=True):
    bias = not bn
    layers = []
    layers.append(
        nn.Conv2DTranspose(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,
                           bias_attr=bias))
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))

    if relu:
        layers.append(nn.LeakyReLU(0.2))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.sublayers(include_self=True):
        if isinstance(m, nn.Conv2DTranspose):
            normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            ones_(m.weight)
            zeros_(m.bias)
    return layers


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose([0, 2, 1])
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = paddle.matmul(R, T)
    else:
        M = paddle.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    t = translation_vector.reshape([-1, 3, 1])
    gather_object = paddle.stack([
        paddle.zeros([
            translation_vector.shape[0],
        ], paddle.float32),
        paddle.ones([
            translation_vector.shape[0],
        ], paddle.float32),
        paddle.squeeze(t[:, 0], axis=-1),
        paddle.squeeze(t[:, 1], axis=-1),
        paddle.squeeze(t[:, 2], axis=-1),
    ])
    gather_index = paddle.to_tensor([
        [1],
        [0],
        [0],
        [2],
        [0],
        [1],
        [0],
        [3],
        [0],
        [0],
        [1],
        [4],
        [0],
        [0],
        [0],
        [1],
    ])
    T = paddle.gather_nd(gather_object, gather_index)
    T = T.reshape([4, 4, -1]).transpose((2, 0, 1))
    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = paddle.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = paddle.cos(angle)
    sa = paddle.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    gather_object = paddle.stack([
        paddle.squeeze(x * xC + ca, axis=(-1, -2)),
        paddle.squeeze(xyC - zs, axis=(-1, -2)),
        paddle.squeeze(zxC + ys, axis=(-1, -2)),
        paddle.squeeze(xyC + zs, axis=(-1, -2)),
        paddle.squeeze(y * yC + ca, axis=(-1, -2)),
        paddle.squeeze(yzC - xs, axis=(-1, -2)),
        paddle.squeeze(zxC - ys, axis=(-1, -2)),
        paddle.squeeze(yzC + xs, axis=(-1, -2)),
        paddle.squeeze(z * zC + ca, axis=(-1, -2)),
        paddle.ones([
            vec.shape[0],
        ], dtype=paddle.float32),
        paddle.zeros([
            vec.shape[0],
        ], dtype=paddle.float32)
    ])
    gather_index = paddle.to_tensor([
        [0],
        [1],
        [2],
        [10],
        [3],
        [4],
        [5],
        [10],
        [6],
        [7],
        [8],
        [10],
        [10],
        [10],
        [10],
        [9],
    ])
    rot = paddle.gather_nd(gather_object, gather_index)
    rot = rot.reshape([4, 4, -1]).transpose((2, 0, 1))
    return rot


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = paddle.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = paddle.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = paddle.mean(paddle.abs(img[:, :, :, :-1] - img[:, :, :, 1:]),
                             1,
                             keepdim=True)
    grad_img_y = paddle.mean(paddle.abs(img[:, :, :-1, :] - img[:, :, 1:, :]),
                             1,
                             keepdim=True)

    grad_disp_x *= paddle.exp(-grad_img_x)
    grad_disp_y *= paddle.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias_attr=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias_attr=False)


def resnet_multiimage_input(num_layers, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]

    block_type = {18: BasicBlock, 50: Bottleneck}[num_layers]

    model = ResNetMultiImageInput(block_type,
                                  num_layers,
                                  blocks,
                                  num_input_images=num_input_images)
    model.init_weights()
    return model


class ConvBlock(nn.Layer):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Layer):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.Pad2D(1, mode='reflect')
        else:
            self.pad = nn.Pad2D(1)
        self.conv = nn.Conv2D(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Layer):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width),
                               range(self.height),
                               indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = self.create_parameter(shape=list(id_coords.shape),
                                               dtype=paddle.float32)
        self.id_coords.set_value(id_coords)
        self.add_parameter("id_coords", self.id_coords)
        self.id_coords.stop_gradient = True

        self.ones = self.create_parameter(
            shape=[self.batch_size, 1, self.height * self.width],
            default_initializer=ones_)
        self.add_parameter("ones", self.ones)
        self.ones.stop_gradient = True

        pix_coords = paddle.unsqueeze(
            paddle.stack([
                self.id_coords[0].reshape([
                    -1,
                ]), self.id_coords[1].reshape([
                    -1,
                ])
            ], 0), 0)
        pix_coords = pix_coords.tile([batch_size, 1, 1])
        pix_coords = paddle.concat([pix_coords, self.ones], 1)
        self.pix_coords = self.create_parameter(shape=list(pix_coords.shape), )
        self.pix_coords.set_value(pix_coords)
        self.add_parameter("pix_coords", self.pix_coords)
        self.pix_coords.stop_gradient = True

    def forward(self, depth, inv_K):
        cam_points = paddle.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.reshape([self.batch_size, 1, -1]) * cam_points
        cam_points = paddle.concat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Layer):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = paddle.matmul(K, T)[:, :3, :]

        cam_points = paddle.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) +
                                             self.eps)
        pix_coords = pix_coords.reshape(
            [self.batch_size, 2, self.height, self.width])
        pix_coords = pix_coords.transpose([0, 2, 3, 1])
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class SSIM(nn.Layer):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2D(3, 1, exclusive=False)
        self.mu_y_pool = nn.AvgPool2D(3, 1, exclusive=False)
        self.sig_x_pool = nn.AvgPool2D(3, 1, exclusive=False)
        self.sig_y_pool = nn.AvgPool2D(3, 1, exclusive=False)
        self.sig_xy_pool = nn.AvgPool2D(3, 1, exclusive=False)

        self.refl = nn.Pad2D(1, mode='reflect')

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return paddle.clip((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ResNetMultiImageInput(ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pypaddle/vision/blob/master/paddlevision/models/resnet.py
    """
    def __init__(self, block, depth, layers, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, depth)
        self.inplanes = 64
        self.conv1 = nn.Conv2D(num_input_images * 3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def init_weights(self):
        for layer in self.sublayers(include_self=True):
            if isinstance(layer, nn.Conv2D):
                kaiming_normal_(layer.weight,
                                mode='fan_out',
                                nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2D):
                ones_(layer.weight)
                zeros_(layer.bias)


class ConvBNLayer(nn.Layer):
    """Conv2D and BatchNorm2D layer.

    Args:
        in_channels (int): Number of channels for the input.
        out_channels (int): Number of channels for the output.
        kernel_size (int): Kernel size.
        stride (int): Stride in the Conv2D layer. Default: 1.
        groups (int): Groups in the Conv2D, Default: 1.
        act (str): Indicate activation after BatchNorm2D layer.
        name (str): the name of an instance of ConvBNLayer.

    Note: weight and bias initialization include initialize values
    and name the restored parameters, values initialization
    are explicit declared in the ```init_weights``` method.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2D(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - 1) // 2,
                            groups=groups,
                            bias_attr=False)

        self._act = act

        self._batch_norm = BatchNorm2D(out_channels)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._act:
            y = getattr(paddle.nn.functional, self._act)(y)
        return y


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DepthDecoder(nn.Layer):
    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=1,
                 use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i +
                                                                           1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s],
                                                  self.num_output_channels)

        self.decoder = nn.LayerList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = paddle.concat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv",
                                                                i)](x))
        return outputs


class PoseDecoder(nn.Layer):
    def __init__(self,
                 num_ch_enc,
                 num_input_features,
                 num_frames_to_predict_for=None,
                 stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2D(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2D(num_input_features * 256, 256, 3,
                                            stride, 1)
        self.convs[("pose", 1)] = nn.Conv2D(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2D(256, 6 * num_frames_to_predict_for,
                                            1)

        self.relu = nn.ReLU()

        self.net = nn.LayerList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [
            self.relu(self.convs["squeeze"](f)) for f in last_features
        ]
        cat_features = paddle.concat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape([-1, self.num_frames_to_predict_for, 1, 6])

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation


class ResnetEncoder(nn.Layer):
    """Pypaddle module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained=False, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
            18: paddle.vision.models.resnet18,
            34: paddle.vision.models.resnet34,
            50: paddle.vision.models.resnet50,
            101: paddle.vision.models.resnet101,
            152: paddle.vision.models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained,
                                                   num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        ######################################
        # night public first conv
        ######################################
        self.conv1 = nn.Conv2D(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()  # NOTE

        self.conv_shared = nn.Conv2D(512, 64, kernel_size=1)

        ##########################################
        # private source encoder, day
        ##########################################
        self.encoder_day = resnets[num_layers](pretrained)
        self.conv_diff_day = nn.Conv2D(
            512, 64, kernel_size=1)  # no bn after conv, so bias=true

        ##########################################
        # private target encoder, night
        ##########################################
        self.encoder_night = resnets[num_layers](pretrained)
        self.conv_diff_night = nn.Conv2D(512, 64, kernel_size=1)

        ######################################
        # shared decoder (small decoder), use a simple de-conv to upsample the features with no skip connection
        ######################################
        self.convt5 = convt_bn_relu(in_channels=512,
                                    out_channels=256,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=256,
                                    out_channels=128,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=128,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=64,
                                    out_channels=64,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1)
        self.convtf = nn.Conv2D(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, input_image, is_night):
        if self.training:
            result = []
            input_data = (input_image - 0.45) / 0.225
            if is_night == 'day':
                # source private encoder, day
                private_feature = self.encoder_day.conv1(input_data)
                private_feature = self.encoder_day.bn1(private_feature)
                private_feature = self.encoder_day.relu(private_feature)
                private_feature = self.encoder_day.maxpool(private_feature)
                private_feature = self.encoder_day.layer1(private_feature)
                private_feature = self.encoder_day.layer2(private_feature)
                private_feature = self.encoder_day.layer3(private_feature)
                private_feature = self.encoder_day.layer4(private_feature)
                private_code = self.conv_diff_day(private_feature)
                private_gram = gram_matrix(private_feature)
                result.append(private_code)
                result.append(private_gram)

            elif is_night == 'night':
                # target private encoder, night
                private_feature = self.encoder_night.conv1(input_data)
                private_feature = self.encoder_night.bn1(private_feature)
                private_feature = self.encoder_night.relu(private_feature)
                private_feature = self.encoder_night.maxpool(private_feature)
                private_feature = self.encoder_night.layer1(private_feature)
                private_feature = self.encoder_night.layer2(private_feature)
                private_feature = self.encoder_night.layer3(private_feature)
                private_feature = self.encoder_night.layer4(private_feature)
                private_code = self.conv_diff_night(private_feature)

                private_gram = gram_matrix(private_feature)
                result.append(private_code)
                result.append(private_gram)

        # shared encoder
        self.features = []
        x = (input_image - 0.45) / 0.225
        if is_night == 'day':
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            self.features.append(self.encoder.relu(x))
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            self.features.append(self.relu(x))

        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        if self.training:
            shared_code = self.conv_shared(self.features[-1])
            shared_gram = gram_matrix(self.features[-1])
            result.append(shared_code)  # use this to calculate loss of diff
            result.append(shared_gram)
            result.append(
                self.features[-1])  # use this to calculate loss of similarity

            union_code = private_feature + self.features[-1]
            rec_code = self.convt5(union_code)
            rec_code = self.convt4(rec_code)
            rec_code = self.convt3(rec_code)
            rec_code = self.convt2(rec_code)
            rec_code = self.convt1(rec_code)
            rec_code = self.convtf(rec_code)
            result.append(rec_code)

            return self.features, result
        else:
            return self.features


class ResnetEncoder_pose(nn.Layer):
    """Pypaddle module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained=False, num_input_images=1):
        super(ResnetEncoder_pose, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {
            18: paddle.vision.models.resnet18,
            34: paddle.vision.models.resnet34,
            50: paddle.vision.models.resnet50,
            101: paddle.vision.models.resnet101,
            152: paddle.vision.models.resnet152
        }

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


@BACKBONES.register()
class ADDS_DepthNet(nn.Layer):
    def __init__(self,
                 num_layers=18,
                 frame_ids=[0, -1, 1],
                 height=256,
                 width=512,
                 batch_size=6,
                 pose_model_input="pairs",
                 use_stereo=False,
                 only_depth_encoder=False,
                 pretrained=None,
                 scales=[0, 1, 2, 3],
                 min_depth=0.1,
                 max_depth=100.0,
                 pose_model_type='separate_resnet',
                 v1_multiscale=False,
                 predictive_mask=False,
                 disable_automasking=False):
        super(ADDS_DepthNet, self).__init__()
        self.num_layers = num_layers
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.frame_ids = frame_ids
        self.pose_model_input = pose_model_input
        self.use_stereo = use_stereo
        self.only_depth_encoder = only_depth_encoder
        self.pretrained = pretrained
        self.scales = scales
        self.pose_model_type = pose_model_type
        self.predictive_mask = predictive_mask
        self.disable_automasking = disable_automasking
        self.v1_multiscale = v1_multiscale
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.num_input_frames = len(self.frame_ids)
        self.num_pose_frames = 2 if self.pose_model_input == "pairs" else self.num_input_frames

        assert self.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.use_stereo and self.frame_ids == [0])

        self.encoder = ResnetEncoder(self.num_layers)
        if not self.only_depth_encoder:
            self.depth = DepthDecoder(self.encoder.num_ch_enc, self.scales)
        if self.use_pose_net and not self.only_depth_encoder:
            if self.pose_model_type == "separate_resnet":
                self.pose_encoder = ResnetEncoder_pose(
                    self.num_layers, num_input_images=self.num_pose_frames)
                self.pose = PoseDecoder(self.pose_encoder.num_ch_enc,
                                        num_input_features=1,
                                        num_frames_to_predict_for=2)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.height // (2**scale)
            w = self.width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(
                self.batch_size, h, w)
            self.project_3d[scale] = Project3D(batch_size, h, w)

    def init_weights(self):
        """First init model's weight"""
        for m in self.sublayers(include_self=True):
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    uniform_ = paddle.nn.initializer.Uniform(-bound, bound)
                    uniform_(m.bias)
        """Second, if provide pretrained ckpt, load it"""
        if self.pretrained:  # load pretrained weights
            load_ckpt(self, self.pretrained)

    def forward(self, inputs, day_or_night='day'):
        if self.training:
            features, result = self.encoder(inputs["color_aug", 0, 0], 'day')
            features_night, result_night = self.encoder(
                inputs[("color_n_aug", 0, 0)], 'night')

            outputs = self.depth(features)
            outputs_night = self.depth(features_night)
            if self.use_pose_net and not self.only_depth_encoder:
                outputs.update(self.predict_poses(inputs, 'day'))
                outputs_night.update(self.predict_poses(inputs, 'night'))

                self.generate_images_pred(inputs, outputs, 'day')
                self.generate_images_pred(inputs, outputs_night, 'night')

            outputs['frame_ids'] = self.frame_ids
            outputs['scales'] = self.scales
            outputs['result'] = result
            outputs['result_night'] = result_night
            outputs_night['frame_ids'] = self.frame_ids
            outputs_night['scales'] = self.scales
            outputs['outputs_night'] = outputs_night
        else:
            if isinstance(inputs, dict):
                input_color = inputs[("color", 0, 0)]
                features = self.encoder(input_color, day_or_night[0])
                outputs = self.depth(features)

                pred_disp, _ = disp_to_depth(outputs[("disp", 0)],
                                             self.min_depth, self.max_depth)

                pred_disp = pred_disp[:, 0].numpy()

                outputs['pred_disp'] = np.squeeze(pred_disp)

                outputs['gt'] = np.squeeze(inputs['depth_gt'].numpy())
            else:
                input_color = inputs
                features = self.encoder(input_color, day_or_night)
                outputs = self.depth(features)

                pred_disp, _ = disp_to_depth(outputs[("disp", 0)],
                                             self.min_depth, self.max_depth)

                pred_disp = pred_disp[:, 0]
                outputs = paddle.squeeze(pred_disp)
        return outputs

    def predict_poses(self, inputs, is_night):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            if is_night:
                pose_feats = {
                    f_i: inputs["color_n_aug", f_i, 0]
                    for f_i in self.frame_ids
                }
            else:
                pose_feats = {
                    f_i: inputs["color_aug", f_i, 0]
                    for f_i in self.frame_ids
                }

            for f_i in self.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.pose_model_type == "separate_resnet":
                        pose_inputs = [
                            self.pose_encoder(paddle.concat(pose_inputs,
                                                            axis=1))
                        ]

                    axisangle, translation = self.pose(pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0,
                             f_i)] = transformation_from_parameters(
                                 axisangle[:, 0],
                                 translation[:, 0],
                                 invert=(f_i < 0))
            return outputs

    def generate_images_pred(self, inputs, outputs, is_night):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        _, _, height, width = inputs['color', 0, 0].shape
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            if self.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [height, width],
                                     mode="bilinear",
                                     align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth
            for i, frame_id in enumerate(self.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                if is_night:
                    inputs[("color_n", frame_id,
                            source_scale)].stop_gradient = False
                    outputs[("color", frame_id,
                             scale)] = paddle.nn.functional.grid_sample(
                                 inputs[("color_n", frame_id, source_scale)],
                                 outputs[("sample", frame_id, scale)],
                                 padding_mode="border",
                                 align_corners=False)

                else:
                    inputs[("color", frame_id,
                            source_scale)].stop_gradient = False
                    outputs[("color", frame_id,
                             scale)] = paddle.nn.functional.grid_sample(
                                 inputs[("color", frame_id, source_scale)],
                                 outputs[("sample", frame_id, scale)],
                                 padding_mode="border",
                                 align_corners=False)

                if not self.disable_automasking:
                    if is_night:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color_n", frame_id, source_scale)]
                    else:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]
