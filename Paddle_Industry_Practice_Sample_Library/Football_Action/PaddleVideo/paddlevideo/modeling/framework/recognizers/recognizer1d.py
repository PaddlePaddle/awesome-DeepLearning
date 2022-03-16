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

from ...registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register()
class Recognizer1D(BaseRecognizer):
    """1D recognizer model framework."""
    def forward_net(self, imgs):
        """Define how the model is going to train, from input to output.
        """
        lstm_logit, lstm_output = self.head(imgs)
        return lstm_logit, lstm_output

    def train_step(self, data_batch):
        """Training step.
        """
        rgb_data, rgb_len, rgb_mask, audio_data, audio_len, audio_mask, labels = data_batch
        imgs = [(rgb_data, rgb_len, rgb_mask),
                (audio_data, audio_len, audio_mask)]

        # call forward
        lstm_logit, lstm_output = self.forward_net(imgs)
        loss = self.head.loss(lstm_logit, labels)
        hit_at_one, perr, gap = self.head.metric(lstm_output, labels)
        loss_metrics = dict()
        loss_metrics['loss'] = loss
        loss_metrics['hit_at_one'] = hit_at_one
        loss_metrics['perr'] = perr
        loss_metrics['gap'] = gap

        return loss_metrics

    def val_step(self, data_batch):
        """Validating setp.
        """
        return self.train_step(data_batch)

    def test_step(self, data_batch):
        """Testing setp.
        """
        return self.train_step(data_batch)

    def infer_step(self, data_batch):
        """Infering setp.
        """
        rgb_data, rgb_len, rgb_mask, audio_data, audio_len, audio_mask = data_batch
        imgs = [(rgb_data, rgb_len, rgb_mask),
                (audio_data, audio_len, audio_mask)]
        # call forward
        lstm_logit, _ = self.forward_net(imgs)
        return lstm_logit
