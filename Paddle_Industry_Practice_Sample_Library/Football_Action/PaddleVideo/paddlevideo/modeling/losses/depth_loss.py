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

import paddle
import paddle.nn as nn

from ..registry import LOSSES
from .base import BaseWeightedLoss


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


class DiffLoss(nn.Layer):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.shape[0]
        input1 = input1.reshape([batch_size, -1])
        input2 = input2.reshape([batch_size, -1])

        input1_l2 = input1
        input2_l2 = input2

        diff_loss = 0
        dim = input1.shape[1]
        for i in range(input1.shape[0]):
            diff_loss = diff_loss + paddle.mean(
                ((input1_l2[i:i + 1, :].mm(input2_l2[i:i + 1, :].T)).pow(2)) /
                dim)

        diff_loss = diff_loss / input1.shape[0]

        return diff_loss


class MSE(nn.Layer):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = paddle.add(real, -pred)
        n = paddle.numel(diffs)
        mse = paddle.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Layer):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = paddle.add(real, -pred)
        n = paddle.numel(diffs)
        simse = paddle.sum(diffs).pow(2) / (n**2)

        return simse


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


@LOSSES.register()
class ADDSLoss(BaseWeightedLoss):
    def __init__(self, avg_reprojection, disparity_smoothness, no_ssim):
        super(ADDSLoss, self).__init__()
        self.avg_reprojection = avg_reprojection
        self.disparity_smoothness = disparity_smoothness
        self.no_ssim = no_ssim

        self.loss_diff = DiffLoss()
        self.loss_recon1 = MSE()
        self.loss_recon2 = SIMSE()
        self.loss_similarity = MSE()

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = paddle.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if not self.no_ssim:
            self.ssim = SSIM()

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs, is_night):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in outputs['scales']:
            loss = 0
            reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", scale)]
            if is_night:
                color = inputs[("color_n", 0, scale)]
                target = inputs[("color_n", 0, source_scale)]
            else:
                color = inputs[("color", 0, scale)]
                target = inputs[("color", 0, source_scale)]

            for frame_id in outputs['frame_ids'][1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            reprojection_losses = paddle.concat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in outputs['frame_ids'][1:]:
                if is_night:
                    pred = inputs[("color_n", frame_id, source_scale)]
                else:
                    pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = paddle.concat(
                identity_reprojection_losses, 1)

            if self.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(
                    1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # add random numbers to break ties
            identity_reprojection_loss = identity_reprojection_loss + paddle.randn(
                identity_reprojection_loss.shape) * 0.00001

            combined = paddle.concat(
                (identity_reprojection_loss, reprojection_loss), axis=1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise = paddle.min(combined, axis=1)

            loss = loss + to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss = loss + self.disparity_smoothness * smooth_loss / (2**scale)
            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= len(outputs['scales'])
        losses["loss"] = total_loss
        return losses

    def forward(self, inputs, outputs):

        losses_day = self.compute_losses(inputs, outputs, 'day')
        losses_night = self.compute_losses(inputs, outputs['outputs_night'],
                                           'night')

        loss = 0
        losses = []
        # diff
        target_diff1 = 0.5 * self.loss_diff(
            outputs['result'][0], outputs['result'][2])  # 10 when batchsize=1
        target_diff2 = 0.5 * self.loss_diff(outputs['result_night'][0],
                                            outputs['result_night'][2])
        losses.append(target_diff1)
        losses.append(target_diff2)
        loss = loss + target_diff1
        loss = loss + target_diff2

        target_diff3 = 1 * self.loss_diff(
            outputs['result'][1], outputs['result'][3])  # 10 when batchsize=1
        target_diff4 = 1 * self.loss_diff(outputs['result_night'][1],
                                          outputs['result_night'][3])
        losses.append(target_diff3)
        losses.append(target_diff4)
        loss = loss + target_diff3
        loss = loss + target_diff4

        # recon
        target_mse = 1 * self.loss_recon1(outputs['result'][5],
                                          inputs["color_aug", 0, 0])
        loss = loss + target_mse

        target_simse = 1 * self.loss_recon2(outputs['result'][5],
                                            inputs["color_aug", 0, 0])
        loss = loss + target_simse

        losses.append(target_mse)
        losses.append(target_simse)
        target_mse_night = 1 * self.loss_recon1(outputs['result_night'][5],
                                                inputs["color_n_aug", 0, 0])
        loss = loss + target_mse_night

        target_simse_night = 1 * self.loss_recon2(outputs['result_night'][5],
                                                  inputs["color_n_aug", 0, 0])
        loss = loss + target_simse_night

        losses.append(target_mse_night)
        losses.append(target_simse_night)

        # depth loss
        pseudo_label = outputs[("disp", 0)].detach()
        depth_loss = 1 * self.loss_similarity(
            outputs['outputs_night'][("disp", 0)], pseudo_label)
        loss = loss + depth_loss

        losses.append(depth_loss)

        outputs['loss'] = loss + losses_day['loss'] + losses_night['loss']
        outputs['losses_day'] = losses_day['loss']
        outputs['losses_night'] = losses_night['loss']

        return outputs
