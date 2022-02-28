# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as functional
import random
from paddle import ParamAttr

from ..registry import BACKBONES


class OctConv3D(nn.Layer):
    def __init__(self, in_filters, filters, kernel_size=3, dilation_rate=(1, 1, 1), alpha=0.25,
                 use_bias=True, kernel_initializer=nn.initializer.KaimingNormal()):
        super(OctConv3D, self).__init__()

        self.low_channels = int(filters * alpha)
        self.high_channels = filters - self.low_channels

        self.high_to_high = nn.Conv3D(in_filters, self.high_channels, kernel_size=kernel_size,
                                      dilation=dilation_rate, padding=(dilation_rate[0], 1, 1),
                                      weight_attr=ParamAttr(initializer=kernel_initializer),
                                      bias_attr=ParamAttr(
                                          initializer=nn.initializer.Constant(value=0.)) if use_bias else use_bias)
        self.high_to_low = nn.Conv3D(self.high_channels, self.low_channels, kernel_size=kernel_size,
                                     dilation=dilation_rate, padding=(dilation_rate[0], 1, 1),
                                     weight_attr=ParamAttr(initializer=kernel_initializer),
                                     bias_attr=False)
        self.low_to_high = nn.Conv3D(in_filters, self.high_channels, kernel_size=kernel_size,
                                     dilation=dilation_rate, padding=(dilation_rate[0], 1, 1),
                                     weight_attr=ParamAttr(initializer=kernel_initializer),
                                     bias_attr=False)
        self.low_to_low = nn.Conv3D(self.high_channels, self.low_channels, kernel_size=kernel_size,
                                    dilation=dilation_rate, padding=(dilation_rate[0], 1, 1),
                                    weight_attr=ParamAttr(initializer=kernel_initializer),
                                    bias_attr=ParamAttr(
                                        initializer=nn.initializer.Constant(value=0.)) if use_bias else use_bias)
        self.upsampler = nn.Upsample(size=(1, 2, 2), data_format='NCDHW')
        self.downsampler = nn.AvgPool3D(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

    @staticmethod
    def pad_to(tensor, target_shape):
        shape = tensor.shape
        padding = [[0, tar - curr] for curr, tar in zip(shape, target_shape)]
        return functional.pad(tensor, padding, "CONSTANT", data_format='NCDHW')

    @staticmethod
    def crop_to(tensor, target_width, target_height):
        return tensor[:, :, :target_height, :target_width]

    def forward(self, inputs):
        low_inputs, high_inputs = inputs

        high_to_high = self.high_to_high(high_inputs)
        high_to_low = self.high_to_low(self.downsampler(high_inputs))

        low_to_high = self.upsampler(self.low_to_high(low_inputs))
        low_to_low = self.low_to_low(low_inputs)

        high_output = high_to_high[:, :, :, :low_to_high.shape[3], :low_to_high.shape[4]] + low_to_high
        low_output = low_to_low + high_to_low[:, :, :, :low_to_low.shape[3], :low_to_low.shape[4]]

        return low_output, high_output


class Conv3DConfigurable(nn.Layer):
    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 octave=False,
                 use_bias=True):
        super(Conv3DConfigurable, self).__init__()
        assert not (separable and octave)

        if separable:
            conv1 = nn.Conv3D(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1),
                              weight_attr=ParamAttr(initializer=nn.initializer.KaimingNormal()),
                              bias_attr=False)
            conv2 = nn.Conv3D(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0),
                              weight_attr=ParamAttr(initializer=nn.initializer.KaimingNormal()),
                              bias_attr=ParamAttr(
                                  initializer=nn.initializer.Constant(value=0.)) if use_bias else use_bias)
            self.layers = nn.LayerList([conv1, conv2])
        elif octave:
            conv = OctConv3D(in_filters, filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                             use_bias=use_bias,
                             kernel_initializer=nn.initializer.KaimingNormal())
            self.layers = [conv]
        else:
            conv = nn.Conv3D(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1),
                             weight_attr=ParamAttr(initializer=nn.initializer.KaimingNormal()),
                             bias_attr=ParamAttr(
                                 initializer=nn.initializer.Constant(value=0.)) if use_bias else use_bias)
            self.layers = nn.LayerList([conv])

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DilatedDCNNV2(nn.Layer):
    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None,
                 octave_conv=False):
        super(DilatedDCNNV2, self).__init__()
        assert not (octave_conv and batch_norm)

        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm, octave=octave_conv)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm, octave=octave_conv)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm, octave=octave_conv)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm, octave=octave_conv)
        self.octave = octave_conv

        self.bn = nn.BatchNorm3D(filters * 4, momentum=0.99, epsilon=1e-03,
                                 weight_attr=ParamAttr(initializer=nn.initializer.Constant(value=1.)),
                                 bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))
                                 ) if batch_norm else None
        self.activation = activation

    def forward(self, inputs):
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        # shape of convi[j]/convi is [B, 3, T, H, W], concat in channel dimension
        if self.octave:
            x = [paddle.concat([conv1[0], conv2[0], conv3[0], conv4[0]], axis=1),
                 paddle.concat([conv1[1], conv2[1], conv3[1], conv4[1]], axis=1)]
        else:
            x = paddle.concat([conv1, conv2, conv3, conv4], axis=1)

        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            if self.octave:
                x = [self.activation(x[0]), self.activation(x[1])]
            else:
                x = self.activation(x)
        return x


class StackedDDCNNV2(nn.Layer):
    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 use_octave_conv=False,
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()
        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        self.DDCNN = nn.LayerList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3D(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3D(kernel_size=(1, 2, 2))
        self.octave = use_octave_conv
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs):
        x = inputs
        shortcut = None

        if self.octave:
            x = [self.pool(x), x]
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
        # shape of x[i] is [B, 3, T, H, W], concat in channel dimension
        if self.octave:
            x = paddle.concat([x[0], self.pool(x[1])], axis=1)

        x = functional.relu(x)

        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        if not self.octave:
            x = self.pool(x)
        return x


class ResNetBlock(nn.Layer):
    def __init__(self, in_filters, filters, strides=(1, 1)):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2D(in_filters, filters, kernel_size=(3, 3), stride=strides, padding=(1, 1),
                               weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(filters,
                                  weight_attr=ParamAttr(initializer=nn.initializer.Constant(value=1.)),
                                  bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))

        self.conv2 = nn.Conv2D(filters, filters, kernel_size=(3, 3), padding=(1, 1),
                               weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(filters,
                                  weight_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)),
                                  bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = inputs
        x += shortcut

        return functional.relu(x)


class ResNetFeatures(nn.Layer):
    def __init__(self, in_filters=3,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(ResNetFeatures, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=in_filters, out_channels=64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3),
                               weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                               bias_attr=False)
        self.bn1 = nn.BatchNorm2D(num_features=64, momentum=0.99, epsilon=1e-03,
                                  weight_attr=ParamAttr(initializer=nn.initializer.Constant(value=1.)),
                                  bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))
                                  )
        self.max_pool = nn.MaxPool2D(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.layer2a = ResNetBlock(64, 64)
        self.layer2b = ResNetBlock(64, 64)

        self.mean = paddle.to_tensor(mean)
        self.std = paddle.to_tensor(std)

    def forward(self, inputs):
        shape = inputs.shape
        x = paddle.reshape(inputs, [shape[0] * shape[2], shape[1], shape[3], shape[4]])
        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x)
        x = functional.relu(x)
        x = self.max_pool(x)
        x = self.layer2a(x)
        x = self.layer2b(x)

        new_shape = x.shape
        x = paddle.reshape(x, [shape[0], new_shape[1], shape[2], new_shape[2], new_shape[3]])
        return x


class FrameSimilarity(nn.Layer):
    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,
                 use_bias=False):
        super(FrameSimilarity, self).__init__()
        self.projection = nn.Linear(in_filters, similarity_dim,
                                    weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                    bias_attr=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim,
                            weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                            bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))

        self.lookup_window = lookup_window
        self.stop_gradient = stop_gradient
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        x = paddle.concat([paddle.mean(x, axis=[3, 4]) for x in inputs], axis=1)
        x = paddle.transpose(x, (0, 2, 1))

        if self.stop_gradient:
            x = x.stop_gradient

        x = self.projection(x)
        x = functional.normalize(x, p=2, axis=2)
        batch_size = paddle.slice(x.shape, starts=[0], ends=[1], axes=[0]) if x.shape[0] == -1 else x.shape[0]
        time_window = x.shape[1]
        similarities = paddle.bmm(x, x.transpose([0, 2, 1]))  # [batch_size, time_window, time_window]

        similarities_padded = functional.pad(similarities,
                                             [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2],
                                             data_format='NCL')

        batch_indices = paddle.arange(0, batch_size).reshape([batch_size, 1, 1])
        batch_indices = paddle.tile(batch_indices, [1, time_window, self.lookup_window])
        time_indices = paddle.arange(0, time_window).reshape([1, time_window, 1])
        time_indices = paddle.tile(time_indices, [batch_size, 1, self.lookup_window])
        lookup_indices = paddle.arange(0, self.lookup_window).reshape([1, 1, self.lookup_window])
        lookup_indices = paddle.tile(lookup_indices, [batch_size, time_window, 1]) + time_indices
        indices = paddle.stack([batch_indices, time_indices, lookup_indices], -1)
        similarities = paddle.gather_nd(similarities_padded, indices)
        return functional.relu(self.fc(similarities))


class ConvexCombinationRegularization(nn.Layer):
    def __init__(self, in_filters, filters=32, delta_scale=10., loss_weight=0.01):
        super(ConvexCombinationRegularization, self).__init__()

        self.projection = nn.Conv3D(in_filters, filters, kernel_size=1, dilation=1, padding=(0, 0, 0),
                                    weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                    bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))
        self.features = nn.Conv3D((filters * 3), filters * 2,
                                  kernel_size=(3, 3, 3), dilation=1, padding=(1, 1, 1),
                                  weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                  bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))
        self.dense = nn.Linear(64, 1, weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()), bias_attr=True)
        self.loss = nn.SmoothL1Loss(reduction='none')
        self.delta_scale = delta_scale
        self.loss_weight = loss_weight

    def forward(self, image_inputs, feature_inputs):
        x = feature_inputs
        x = self.projection(x)
        x = functional.relu(x)
        batch_size = x.shape[0]
        window_size = x.shape[2]
        first_frame = paddle.tile(x[:, :, :1], [1, 1, window_size, 1, 1])
        last_frame = paddle.tile(x[:, :, -1:], [1, 1, window_size, 1, 1])
        x = paddle.concat([x, first_frame, last_frame], 1)
        x = self.features(x)
        x = functional.relu(x)
        x = paddle.mean(x, axis=[3, 4])
        x = paddle.transpose(x, (0, 2, 1))
        alpha = self.dense(x)
        alpha = paddle.transpose(alpha, (0, 2, 1))

        first_img = paddle.tile(image_inputs[:, :, :1], [1, 1, window_size, 1, 1])
        last_img = paddle.tile(image_inputs[:, :, -1:], [1, 1, window_size, 1, 1])

        alpha_ = functional.sigmoid(alpha)
        alpha_ = paddle.reshape(alpha_, [batch_size, 1, window_size, 1, 1])
        predictions_ = (alpha_ * first_img + (1 - alpha_) * last_img)
        loss_ = self.loss(label=image_inputs / self.delta_scale, input=predictions_ / self.delta_scale)
        loss_ = self.loss_weight * paddle.mean(loss_)
        return alpha, loss_


class ColorHistograms(nn.Layer):
    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window, output_dim,
                            weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                            bias_attr=ParamAttr(
                                initializer=nn.initializer.Constant(value=0.))) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def compute_color_histograms(self, frames):
        frames = frames.astype('int32')

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R // 32, G // 32, B // 32
            return (R * 64) + (G * 8) + B

        batch_size = paddle.slice(frames.shape, starts=[0], ends=[1], axes=[0]) if frames.shape[0] == -1 else frames.shape[0]
        time_window, height, width, no_channels = frames.shape[1:]

        assert no_channels == 3 or no_channels == 6
        if no_channels == 3:
            frames_flatten = frames.reshape([-1, height * width, 3])
        else:
            frames_flatten = frames.reshape([-1, height * width * 2, 3])

        binned_values = get_bin(frames_flatten)

        frame_bin_prefix = (paddle.arange(0, batch_size * time_window) * 512).reshape([-1, 1])
        binned_values = (binned_values + frame_bin_prefix).reshape([-1, 1])
        histograms = paddle.zeros_like(frame_bin_prefix, dtype='int32').tile([512]).reshape([-1])
        histograms = histograms.scatter_nd_add(binned_values, paddle.ones_like(binned_values, dtype='int32').reshape([-1]))
        histograms = histograms.reshape([batch_size, time_window, 512]).astype('float32')
        histograms_normalized = functional.normalize(histograms, p=2, axis=2)
        return histograms_normalized

    def forward(self, inputs):
        x = self.compute_color_histograms(inputs)
        batch_size = paddle.slice(x.shape, starts=[0], ends=[1], axes=[0]) if x.shape[0] == -1 else x.shape[0]
        time_window = x.shape[1]
        similarities = paddle.bmm(x, x.transpose([0, 2, 1]))  # [batch_size, time_window, time_window]
        similarities_padded = functional.pad(similarities,
                                             [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2],
                                             data_format='NCL')

        batch_indices = paddle.arange(0, batch_size).reshape([batch_size, 1, 1])
        batch_indices = paddle.tile(batch_indices, [1, time_window, self.lookup_window])
        time_indices = paddle.arange(0, time_window).reshape([1, time_window, 1])
        time_indices = paddle.tile(time_indices, [batch_size, 1, self.lookup_window])
        lookup_indices = paddle.arange(0, self.lookup_window).reshape([1, 1, self.lookup_window])
        lookup_indices = paddle.tile(lookup_indices, [batch_size, time_window, 1]) + time_indices

        indices = paddle.stack([batch_indices, time_indices, lookup_indices], -1)
        similarities = paddle.gather_nd(similarities_padded, indices)

        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities


@BACKBONES.register()
class TransNetV2(nn.Layer):
    """TransNetV2 model from
    `"TransNet V2: An effective deep network architecture for fast shot transition detection" <https://arxiv.org/abs/2008.04838>`_
    """
    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,
                 use_resnet_features=False,
                 use_resnet_like_top=False,
                 frame_similarity_on_last_layer=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        super(TransNetV2, self).__init__()

        self.mean = np.array(mean, np.float32).reshape([1, 3, 1, 1]) * 255
        self.std = np.array(std, np.float32).reshape([1, 3, 1, 1]) * 255

        self.use_resnet_features = use_resnet_features
        self.resnet_layers = ResNetFeatures(in_filters=3, mean=self.mean, std=self.std) if self.use_resnet_features else None
        self.resnet_like_top = use_resnet_like_top
        if self.resnet_like_top:
            self.resnet_like_top_conv = nn.Conv3D(64 if self.use_resnet_features else 3, 32, kernel_size=(3, 7, 7),
                                                  stride=(1, 2, 2),
                                                  padding=(1, 3, 3),
                                                  weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                                  bias_attr=False)
            self.resnet_like_top_bn = nn.BatchNorm3D(32, momentum=0.99, epsilon=1e-03,
                                                     weight_attr=ParamAttr(
                                                         initializer=nn.initializer.Constant(value=1.)),
                                                     bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.)))
            self.resnet_like_top_max_pool = nn.MaxPool3D(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                                         padding=(0, 1, 1))

        if self.resnet_like_top:
            in_filters = 32
        elif self.use_resnet_features:
            in_filters = 64
        else:
            in_filters = 3
        self.SDDCNN = nn.LayerList(
            [StackedDDCNNV2(in_filters=in_filters, n_blocks=S, filters=F,
                            stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128,
            use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        self.use_mean_pooling = use_mean_pooling

        self.has_downsample = False
        if self.use_resnet_features or self.resnet_like_top or self.use_mean_pooling:
            self.has_downsample = True
        self.fc1 = nn.Linear(512 if self.has_downsample else output_dim, D,
                             weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                             bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))
                             )
        self.frame_similarity_on_last_layer = frame_similarity_on_last_layer
        self.cls_layer1 = nn.Linear(1152 if self.frame_similarity_on_last_layer else D, 1,
                                    weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                    bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))
                                    )
        self.cls_layer2 = nn.Linear(1152 if self.frame_similarity_on_last_layer else D, 1,
                                    weight_attr=ParamAttr(initializer=nn.initializer.XavierUniform()),
                                    bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))
                                    ) if use_many_hot_targets else None

        self.convex_comb_reg = ConvexCombinationRegularization(
            in_filters=(F * 2 ** (L - 1) * 4)) if use_convex_comb_reg else None

    def forward(self, inputs):
        assert list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == paddle.float32, \
            "incorrect input type and/or shape"
        out_dict = {}

        # shape [B, T, H, W, 3] to shape [B, 3, T, H, W]
        x = inputs.transpose([0, 4, 1, 2, 3])
        if self.use_resnet_features:
            x = self.resnet_layers(x)
        else:
            x = x / 255.
        inputs = inputs.clip(min=0).astype('uint8')
        if self.resnet_like_top:
            x = self.resnet_like_top_conv(x)
            x = self.resnet_like_top_bn(x)
            x = self.resnet_like_top_max_pool(x)
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)
        if self.convex_comb_reg is not None:
            out_dict["alphas"], out_dict["comb_reg_loss"] = self.convex_comb_reg(inputs.transpose([0, 4, 1, 2, 3]), x)
        if self.use_mean_pooling:
            x = paddle.mean(x, axis=[3, 4])
            x = x.transpose([0, 2, 1])
        else:
            x = x.transpose([0, 2, 3, 4, 1])
            x = x.reshape([x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4]])
        if self.frame_sim_layer is not None:
            x = paddle.concat([self.frame_sim_layer(block_features), x], 2)
        if self.color_hist_layer is not None:
            x = paddle.concat([self.color_hist_layer(inputs), x], 2)
        x = self.fc1(x)
        x = functional.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.frame_sim_layer is not None and self.frame_similarity_on_last_layer:
            x = paddle.concat([self.frame_sim_layer(block_features), x], 2)
        one_hot = self.cls_layer1(x)
        if self.cls_layer2 is not None:
            out_dict["many_hot"] = self.cls_layer2(x)

        if len(out_dict) > 0:
            return one_hot, out_dict

        return one_hot

