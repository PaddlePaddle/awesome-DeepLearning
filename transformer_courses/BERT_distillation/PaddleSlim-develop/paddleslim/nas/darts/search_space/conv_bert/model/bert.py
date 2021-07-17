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

import os
import six
import json
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Embedding, LayerNorm, Linear, to_variable, Layer, guard
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import MSRA
from .transformer_encoder import EncoderLayer


class BertModelLayer(Layer):
    def __init__(self,
                 num_labels,
                 emb_size=128,
                 hidden_size=768,
                 n_layer=12,
                 voc_size=30522,
                 max_position_seq_len=512,
                 sent_types=2,
                 return_pooled_out=True,
                 initializer_range=1.0,
                 conv_type="conv_bn",
                 search_layer=False,
                 use_fp16=False,
                 use_fixed_gumbel=False,
                 gumbel_alphas=None):
        super(BertModelLayer, self).__init__()

        self._emb_size = emb_size
        self._hidden_size = hidden_size
        self._n_layer = n_layer
        self._voc_size = voc_size
        self._max_position_seq_len = max_position_seq_len
        self._sent_types = sent_types
        self.return_pooled_out = return_pooled_out

        self.use_fixed_gumbel = use_fixed_gumbel

        self._word_emb_name = "s_word_embedding"
        self._pos_emb_name = "s_pos_embedding"
        self._sent_emb_name = "s_sent_embedding"
        self._dtype = "float16" if use_fp16 else "float32"

        self._conv_type = conv_type
        self._search_layer = search_layer
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=initializer_range)

        self._src_emb = Embedding(
            size=[self._voc_size, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._word_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._pos_emb = Embedding(
            size=[self._max_position_seq_len, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._pos_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._sent_emb = Embedding(
            size=[self._sent_types, self._emb_size],
            param_attr=fluid.ParamAttr(
                name=self._sent_emb_name, initializer=self._param_initializer),
            dtype=self._dtype)

        self._emb_fac = Linear(
            input_dim=self._emb_size,
            output_dim=self._hidden_size,
            param_attr=fluid.ParamAttr(name="s_emb_factorization"))

        self._encoder = EncoderLayer(
            num_labels=num_labels,
            n_layer=self._n_layer,
            hidden_size=self._hidden_size,
            search_layer=self._search_layer,
            use_fixed_gumbel=self.use_fixed_gumbel,
            gumbel_alphas=gumbel_alphas)

    def emb_names(self):
        return self._src_emb.parameters() + self._pos_emb.parameters(
        ) + self._sent_emb.parameters()

    def emb_names(self):
        return self._src_emb.parameters() + self._pos_emb.parameters(
        ) + self._sent_emb.parameters()

    def max_flops(self):
        return self._encoder.max_flops

    def max_model_size(self):
        return self._encoder.max_model_size

    def arch_parameters(self):
        return [self._encoder.alphas]  #, self._encoder.k]

    def forward(self, data_ids, epoch):
        """
        forward
        """
        ids0 = data_ids[5]
        ids1 = data_ids[6]

        src_emb_0 = self._src_emb(ids0)
        src_emb_1 = self._src_emb(ids1)
        emb_out_0 = self._emb_fac(src_emb_0)
        emb_out_1 = self._emb_fac(src_emb_1)
        # (bs, seq_len, hidden_size)

        enc_outputs = self._encoder(emb_out_0, emb_out_1, epoch)

        return enc_outputs
