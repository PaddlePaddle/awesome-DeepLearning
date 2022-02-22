#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""BERT fine-tuning in Paddle Dygraph Mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six
import sys
if six.PY2:
    reload(sys)
    sys.setdefaultencoding('utf8')
import ast
import time
import argparse
import numpy as np
import multiprocessing
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable, Layer, Linear
from paddle.fluid.dygraph.base import to_variable
from .reader.cls import *
from .model.bert import BertModelLayer
from .optimization import Optimizer
from .utils.init import init_from_static_model
from paddleslim.teachers.bert import BERTClassifier

__all__ = ["AdaBERTClassifier"]


class AdaBERTClassifier(Layer):
    def __init__(self,
                 num_labels,
                 n_layer=8,
                 emb_size=768,
                 hidden_size=768,
                 gamma=0.8,
                 beta=4,
                 task_name='mnli',
                 conv_type="conv_bn",
                 search_layer=False,
                 teacher_model=None,
                 data_dir=None,
                 use_fixed_gumbel=False,
                 gumbel_alphas=None,
                 fix_emb=False,
                 t=5.0):
        super(AdaBERTClassifier, self).__init__()
        self._n_layer = n_layer
        self._num_labels = num_labels
        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._gamma = gamma
        self._beta = beta
        self._conv_type = conv_type
        self._search_layer = search_layer
        self._teacher_model = teacher_model
        self._data_dir = data_dir
        self.use_fixed_gumbel = use_fixed_gumbel

        self.T = t
        print(
            "----------------------load teacher model and test----------------------------------------"
        )
        self.teacher = BERTClassifier(
            num_labels, task_name=task_name, model_path=self._teacher_model)
        # global setting, will be overwritten when training(about 1% acc loss)
        self.teacher.eval()
        self.teacher.test(self._data_dir)
        print(
            "----------------------finish load teacher model and test----------------------------------------"
        )
        self.student = BertModelLayer(
            num_labels=num_labels,
            n_layer=self._n_layer,
            emb_size=self._emb_size,
            hidden_size=self._hidden_size,
            conv_type=self._conv_type,
            search_layer=self._search_layer,
            use_fixed_gumbel=self.use_fixed_gumbel,
            gumbel_alphas=gumbel_alphas)

        fix_emb = False
        for s_emb, t_emb in zip(self.student.emb_names(),
                                self.teacher.emb_names()):
            t_emb.stop_gradient = True
            if fix_emb:
                s_emb.stop_gradient = True
            print(
                "Assigning embedding[{}] from teacher to embedding[{}] in student.".
                format(t_emb.name, s_emb.name))
            fluid.layers.assign(input=t_emb, output=s_emb)
            print(
                "Assigned embedding[{}] from teacher to embedding[{}] in student.".
                format(t_emb.name, s_emb.name))

    def forward(self, data_ids, epoch):
        return self.student(data_ids, epoch)

    def arch_parameters(self):
        return self.student.arch_parameters()

    def loss(self, data_ids, epoch):
        labels = data_ids[4]

        s_logits = self.student(data_ids, epoch)

        t_enc_outputs, t_logits, t_losses, t_accs, _ = self.teacher(data_ids)

        #define kd loss
        kd_weights = []
        for i in range(len(s_logits)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(s_logits))))
            kd_weights.append(t_losses[j].numpy())

        kd_weights = np.array(kd_weights)
        kd_weights = np.squeeze(kd_weights)
        kd_weights = to_variable(kd_weights)
        kd_weights = fluid.layers.softmax(-kd_weights)

        kd_losses = []
        for i in range(len(s_logits)):
            j = int(np.ceil(i * (float(len(t_logits)) / len(s_logits))))
            t_logit = t_logits[j]
            s_logit = s_logits[i]
            t_logit.stop_gradient = True
            t_probs = fluid.layers.softmax(t_logit)  # P_j^T
            s_probs = fluid.layers.softmax(s_logit / self.T)  #P_j^S
            #kd_loss = -t_probs * fluid.layers.log(s_probs)
            kd_loss = fluid.layers.cross_entropy(
                input=s_probs, label=t_probs, soft_label=True)
            kd_loss = fluid.layers.reduce_mean(kd_loss)
            kd_loss = fluid.layers.scale(kd_loss, scale=kd_weights[i])
            kd_losses.append(kd_loss)
        kd_loss = fluid.layers.sum(kd_losses)

        losses = []
        for logit in s_logits:
            ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
                logits=logit, label=labels, return_softmax=True)
            loss = fluid.layers.mean(x=ce_loss)
            losses.append(loss)

            num_seqs = fluid.layers.create_tensor(dtype='int64')
            accuracy = fluid.layers.accuracy(
                input=probs, label=labels, total=num_seqs)
        ce_loss = fluid.layers.sum(losses)

        total_loss = (1 - self._gamma) * ce_loss + self._gamma * kd_loss

        return total_loss, accuracy, ce_loss, kd_loss, s_logits
