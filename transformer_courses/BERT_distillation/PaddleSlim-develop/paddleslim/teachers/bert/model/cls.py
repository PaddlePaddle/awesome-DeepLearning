# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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
"dygraph transformer layers"

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Layer

from .bert import BertModelLayer


class ClsModelLayer(Layer):
    """
    classify model
    """

    def __init__(self,
                 config,
                 num_labels,
                 is_training=True,
                 return_pooled_out=True,
                 loss_scaling=1.0,
                 use_fp16=False):
        super(ClsModelLayer, self).__init__()
        self.config = config
        self.is_training = is_training
        self.use_fp16 = use_fp16
        self.loss_scaling = loss_scaling
        self.n_layers = config['num_hidden_layers']
        self.return_pooled_out = return_pooled_out

        self.bert_layer = BertModelLayer(
            config=self.config, return_pooled_out=True, use_fp16=self.use_fp16)

        self.cls_fc = list()
        for i in range(self.n_layers):
            fc = Linear(
                input_dim=self.config["hidden_size"],
                output_dim=num_labels,
                param_attr=fluid.ParamAttr(
                    name="cls_out_%d_w" % i,
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
                bias_attr=fluid.ParamAttr(
                    name="cls_out_%d_b" % i,
                    initializer=fluid.initializer.Constant(0.)))
            fc = self.add_sublayer("cls_fc_%d" % i, fc)
            self.cls_fc.append(fc)

    def emb_names(self):
        return self.bert_layer.emb_names()

    def forward(self, data_ids):
        """
        forward
        """
        src_ids = data_ids[0]
        position_ids = data_ids[1]
        sentence_ids = data_ids[2]
        input_mask = data_ids[3]
        labels = data_ids[4]

        enc_outputs, next_sent_feats = self.bert_layer(
            src_ids, position_ids, sentence_ids, input_mask)

        if not self.return_pooled_out:
            cls_feat = fluid.layers.dropout(
                x=next_sent_feats[-1],
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            logits = self.cls_fc[-1](cls_feat)
            probs = fluid.layers.softmax(logits)
            num_seqs = fluid.layers.create_tensor(dtype='int64')
            accuracy = fluid.layers.accuracy(
                input=probs, label=labels, total=num_seqs)
            return enc_outputs, logits, accuracy, num_seqs

        logits = []
        losses = []
        accuracys = []
        for next_sent_feat, fc in zip(next_sent_feats, self.cls_fc):
            cls_feat = fluid.layers.dropout(
                x=next_sent_feat,
                dropout_prob=0.1,
                dropout_implementation="upscale_in_train")
            logit = fc(cls_feat)
            logits.append(logit)

            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=logit, label=labels, return_softmax=True)
            loss = fluid.layers.mean(x=ce_loss)
            losses.append(loss)

            if self.use_fp16 and self.loss_scaling > 1.0:
                loss *= self.loss_scaling

            num_seqs = fluid.layers.create_tensor(dtype='int64')
            accuracy = fluid.layers.accuracy(
                input=probs, label=labels, total=num_seqs)
            accuracys.append(accuracy)
        total_loss = fluid.layers.sum(losses)

        return total_loss, logits, losses, accuracys, num_seqs
