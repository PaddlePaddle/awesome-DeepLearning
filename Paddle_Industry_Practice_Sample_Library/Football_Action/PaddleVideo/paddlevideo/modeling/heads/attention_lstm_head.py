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
from paddle import ParamAttr
from paddle.nn.initializer import Normal
from paddle.regularizer import L2Decay

from ...metrics.youtube8m import eval_util as youtube8m_metrics
from ..registry import HEADS
from ..weight_init import weight_init_
from .base import BaseHead


@HEADS.register()
class AttentionLstmHead(BaseHead):
    """AttentionLstmHead.
    Args: TODO
    """
    def __init__(self,
                 num_classes=3862,
                 feature_num=2,
                 feature_dims=[1024, 128],
                 embedding_size=512,
                 lstm_size=1024,
                 in_channels=2048,
                 loss_cfg=dict(name='CrossEntropyLoss')):
        super(AttentionLstmHead, self).__init__(num_classes, in_channels,
                                                loss_cfg)
        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.feature_num = len(self.feature_dims)
        for i in range(self.feature_num):  # 0:rgb, 1:audio
            fc_feature = paddle.nn.Linear(in_features=self.feature_dims[i],
                                          out_features=self.embedding_size)
            self.add_sublayer("fc_feature{}".format(i), fc_feature)

            bi_lstm = paddle.nn.LSTM(input_size=self.embedding_size,
                                     hidden_size=self.lstm_size,
                                     direction="bidirectional")
            self.add_sublayer("bi_lstm{}".format(i), bi_lstm)

            drop_rate = 0.5
            self.dropout = paddle.nn.Dropout(drop_rate)

            att_fc = paddle.nn.Linear(in_features=self.lstm_size * 2,
                                      out_features=1)
            self.add_sublayer("att_fc{}".format(i), att_fc)
            self.softmax = paddle.nn.Softmax()

        self.fc_out1 = paddle.nn.Linear(in_features=self.lstm_size * 4,
                                        out_features=8192,
                                        bias_attr=ParamAttr(
                                            regularizer=L2Decay(0.0),
                                            initializer=Normal()))
        self.relu = paddle.nn.ReLU()
        self.fc_out2 = paddle.nn.Linear(in_features=8192,
                                        out_features=4096,
                                        bias_attr=ParamAttr(
                                            regularizer=L2Decay(0.0),
                                            initializer=Normal()))
        self.fc_logit = paddle.nn.Linear(in_features=4096,
                                         out_features=self.num_classes,
                                         bias_attr=ParamAttr(
                                             regularizer=L2Decay(0.0),
                                             initializer=Normal()))
        self.sigmoid = paddle.nn.Sigmoid()

    def init_weights(self):
        pass

    def forward(self, inputs):
        # inputs = [(rgb_data, rgb_len, rgb_mask), (audio_data, audio_len, audio_mask)]
        # deal with features with different length
        # 1. padding to same lenght, make a tensor
        # 2. make a mask tensor with the same shpae with 1
        # 3. compute output using mask tensor, s.t. output is nothing todo with padding
        assert (len(inputs) == self.feature_num
                ), "Input tensor does not contain {} features".format(
                    self.feature_num)
        att_outs = []
        for i in range(len(inputs)):
            # 1. fc
            m = getattr(self, "fc_feature{}".format(i))
            output_fc = m(inputs[i][0])
            output_fc = paddle.tanh(output_fc)

            # 2. bi_lstm
            m = getattr(self, "bi_lstm{}".format(i))
            lstm_out, _ = m(inputs=output_fc, sequence_length=inputs[i][1])

            lstm_dropout = self.dropout(lstm_out)

            # 3. att_fc
            m = getattr(self, "att_fc{}".format(i))
            lstm_weight = m(lstm_dropout)

            # 4. softmax replace start, for it's relevant to sum in time step
            lstm_exp = paddle.exp(lstm_weight)
            lstm_mask = paddle.mean(inputs[i][2], axis=2)
            lstm_mask = paddle.unsqueeze(lstm_mask, axis=2)
            lstm_exp_with_mask = paddle.multiply(x=lstm_exp, y=lstm_mask)
            lstm_sum_with_mask = paddle.sum(lstm_exp_with_mask, axis=1)
            exponent = -1
            lstm_denominator = paddle.pow(lstm_sum_with_mask, exponent)
            lstm_denominator = paddle.unsqueeze(lstm_denominator, axis=2)
            lstm_softmax = paddle.multiply(x=lstm_exp, y=lstm_denominator)
            lstm_weight = lstm_softmax
            # softmax replace end

            lstm_scale = paddle.multiply(x=lstm_dropout, y=lstm_weight)

            # 5. sequence_pool's replace start, for it's relevant to sum in time step
            lstm_scale_with_mask = paddle.multiply(x=lstm_scale, y=lstm_mask)
            fea_lens = inputs[i][1]
            fea_len = int(fea_lens[0])
            lstm_pool = paddle.sum(lstm_scale_with_mask, axis=1)
            # sequence_pool's replace end
            att_outs.append(lstm_pool)
        att_out = paddle.concat(att_outs, axis=1)
        fc_out1 = self.fc_out1(att_out)
        fc_out1_act = self.relu(fc_out1)
        fc_out2 = self.fc_out2(fc_out1_act)
        fc_out2_act = paddle.tanh(fc_out2)
        fc_logit = self.fc_logit(fc_out2_act)
        output = self.sigmoid(fc_logit)
        return fc_logit, output

    def loss(self, lstm_logit, labels, **kwargs):
        labels.stop_gradient = True
        losses = dict()
        bce_logit_loss = paddle.nn.BCEWithLogitsLoss(reduction='sum')
        sum_cost = bce_logit_loss(lstm_logit, labels)
        return sum_cost

    def metric(self, lstm_output, labels):
        pred = lstm_output.numpy()
        label = labels.numpy()
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(
            pred, label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        return hit_at_one, perr, gap
