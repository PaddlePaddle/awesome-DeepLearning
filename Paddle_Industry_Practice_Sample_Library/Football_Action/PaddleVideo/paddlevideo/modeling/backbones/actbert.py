# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import sys
import numpy as np
import math
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import (Conv2D, BatchNorm2D, Linear, Dropout)
from paddle.nn.initializer import Constant, Normal
from ...utils.save_load import load_ckpt
from ..registry import BACKBONES
from ..weight_init import weight_init_

ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": F.swish}


class BertEmbeddings(nn.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, max_position_embeddings, type_vocab_size,
                 hidden_size, hidden_dropout_prob):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size,
                                            hidden_size,
                                            padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.shape[1]
        position_ids = paddle.arange(end=seq_length, dtype="int64")
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)  #8,36  -> 8,36,768
        position_embeddings = self.position_embeddings(
            position_ids)  #8,36  -> 8,36,768
        token_type_embeddings = self.token_type_embeddings(
            token_type_ids)  #8,36  -> 8,36,768

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertImageEmbeddings(nn.Layer):
    def __init__(self, v_feature_size, v_hidden_size, v_hidden_dropout_prob):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(v_feature_size, v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, v_hidden_size)
        self.LayerNorm = nn.LayerNorm(v_hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(v_hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(
            input_ids)  #8,37,2048 -> 8,37,1024
        loc_embeddings = self.image_location_embeddings(
            input_loc)  #8,37,5 -> 8,37,1024
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # shape: bs*seq_len*hs


class BertActionEmbeddings(nn.Layer):
    def __init__(self, a_feature_size, a_hidden_size, a_hidden_dropout_prob):
        super(BertActionEmbeddings, self).__init__()
        self.action_embeddings = nn.Linear(a_feature_size, a_hidden_size)
        self.LayerNorm = nn.LayerNorm(a_hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(a_hidden_dropout_prob)

    def forward(self, input_ids):
        action_embeddings = self.action_embeddings(
            input_ids)  #8,5,2048 -> 8,5,768
        embeddings = self.LayerNorm(action_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Layer):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_probs_dropout_prob):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer,
                                         key_layer.transpose((0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


class BertSelfOutput(nn.Layer):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Layer):
    def __init__(self, hidden_size, hidden_dropout_prob, num_attention_heads,
                 attention_probs_dropout_prob):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads,
                                      attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str) or (sys.version_info[0] == 2
                                           and isinstance(hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Layer):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertEntAttention(nn.Layer):
    """Core mudule of tangled transformer.
    """
    def __init__(
        self,
        hidden_size,
        v_hidden_size,
        a_hidden_size,
        bi_hidden_size,
        attention_probs_dropout_prob,
        v_attention_probs_dropout_prob,
        a_attention_probs_dropout_prob,
        av_attention_probs_dropout_prob,
        at_attention_probs_dropout_prob,
        bi_num_attention_heads,
    ):
        super(BertEntAttention, self).__init__()
        if bi_hidden_size % bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (bi_hidden_size, bi_num_attention_heads))

        self.num_attention_heads = bi_num_attention_heads
        self.attention_head_size = int(bi_hidden_size / bi_num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self attention layers for vision input
        self.query1 = nn.Linear(v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(v_hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(v_attention_probs_dropout_prob)

        # self attention layers for text input
        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)
        self.dropout2 = nn.Dropout(attention_probs_dropout_prob)

        # self attention layers for action input
        self.query3 = nn.Linear(a_hidden_size, self.all_head_size)
        self.key3 = nn.Linear(a_hidden_size, self.all_head_size)
        self.value3 = nn.Linear(a_hidden_size, self.all_head_size)
        self.dropout3 = nn.Dropout(a_attention_probs_dropout_prob)

        # self attention layers for action_text
        self.key_at = nn.Linear(bi_hidden_size, self.all_head_size)
        self.value_at = nn.Linear(bi_hidden_size, self.all_head_size)
        self.dropout_at = nn.Dropout(av_attention_probs_dropout_prob)

        # self attention layers for action_vision
        self.key_av = nn.Linear(bi_hidden_size, self.all_head_size)
        self.value_av = nn.Linear(bi_hidden_size, self.all_head_size)
        self.dropout_av = nn.Dropout(at_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        input_tensor3,
        attention_mask3,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)

        # for action input:
        mixed_query_layer3 = self.query3(input_tensor3)
        mixed_key_layer3 = self.key3(input_tensor3)
        mixed_value_layer3 = self.value3(input_tensor3)

        query_layer3 = self.transpose_for_scores(mixed_query_layer3)
        key_layer3 = self.transpose_for_scores(mixed_key_layer3)
        value_layer3 = self.transpose_for_scores(mixed_value_layer3)

        def do_attention(query_layer, key_layer, value_layer, attention_mask,
                         dropout):
            """ compute attention """
            attention_scores = paddle.matmul(query_layer,
                                             key_layer.transpose((0, 1, 3, 2)))
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(axis=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = dropout(attention_probs)

            context_layer = paddle.matmul(attention_probs, value_layer)
            context_layer = context_layer.transpose((0, 2, 1, 3))
            new_context_layer_shape = context_layer.shape[:-2] + [
                self.all_head_size
            ]
            context_layer = context_layer.reshape(new_context_layer_shape)
            return context_layer

        context_av = do_attention(query_layer3, key_layer1, value_layer1,
                                  attention_mask1, self.dropout_av)
        context_at = do_attention(query_layer3, key_layer2, value_layer2,
                                  attention_mask2, self.dropout_at)

        context_key_av = self.key_av(context_av).transpose((0, 2, 1))
        # interpolate only support 4-D tensor now.
        context_key_av = F.interpolate(context_key_av.unsqueeze(-1),
                                       size=(key_layer2.shape[2],
                                             1)).squeeze(-1)
        context_key_av = self.transpose_for_scores(
            context_key_av.transpose((0, 2, 1)))
        key_layer2 = key_layer2 + context_key_av

        context_key_at = self.key_at(context_at).transpose((0, 2, 1))
        context_key_at = F.interpolate(context_key_at.unsqueeze(-1),
                                       size=(key_layer1.shape[2],
                                             1)).squeeze(-1)
        context_key_at = self.transpose_for_scores(
            context_key_at.transpose((0, 2, 1)))
        key_layer1 = key_layer1 + context_key_at

        context_val_av = self.value_at(context_av).transpose((0, 2, 1))
        context_val_av = F.interpolate(context_val_av.unsqueeze(-1),
                                       size=(value_layer2.shape[2],
                                             1)).squeeze(-1)
        context_val_av = self.transpose_for_scores(
            context_val_av.transpose((0, 2, 1)))
        value_layer2 = value_layer2 + context_val_av

        context_val_at = self.value_at(context_at).transpose((0, 2, 1))
        context_val_at = F.interpolate(context_val_at.unsqueeze(-1),
                                       size=(value_layer1.shape[2],
                                             1)).squeeze(-1)
        context_val_at = self.transpose_for_scores(
            context_val_at.transpose((0, 2, 1)))
        value_layer1 = value_layer1 + context_val_at

        context_layer1 = do_attention(query_layer1, key_layer1, value_layer1,
                                      attention_mask1, self.dropout1)
        context_layer2 = do_attention(query_layer2, key_layer2, value_layer2,
                                      attention_mask2, self.dropout2)
        context_layer3 = do_attention(query_layer3, key_layer3, value_layer3,
                                      attention_mask3, self.dropout3)

        return context_layer1, context_layer2, context_layer3  # vision, text, action


class BertEntOutput(nn.Layer):
    def __init__(
        self,
        bi_hidden_size,
        hidden_size,
        v_hidden_size,
        v_hidden_dropout_prob,
        hidden_dropout_prob,
    ):
        super(BertEntOutput, self).__init__()

        self.dense1 = nn.Linear(bi_hidden_size, v_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(v_hidden_size, epsilon=1e-12)
        self.dropout1 = nn.Dropout(v_hidden_dropout_prob)

        self.dense2 = nn.Linear(bi_hidden_size, hidden_size)
        self.LayerNorm2 = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout2 = nn.Dropout(hidden_dropout_prob)

        self.dense3 = nn.Linear(bi_hidden_size, hidden_size)
        self.LayerNorm3 = nn.LayerNorm(hidden_size, epsilon=1e-12)
        self.dropout3 = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        hidden_states1,
        input_tensor1,
        hidden_states2,
        input_tensor2,
        hidden_states3,
        input_tensor3,
    ):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        context_state3 = self.dense3(hidden_states3)
        context_state3 = self.dropout3(context_state3)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)
        hidden_states3 = self.LayerNorm3(context_state3 + input_tensor3)

        return hidden_states1, hidden_states2, hidden_states3


class BertLayer(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act,
                 hidden_dropout_prob, num_attention_heads,
                 attention_probs_dropout_prob):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, hidden_dropout_prob,
                                       num_attention_heads,
                                       attention_probs_dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size,
                                             hidden_act)
        self.output = BertOutput(intermediate_size, hidden_size,
                                 hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertConnectionLayer(nn.Layer):
    def __init__(self, hidden_size, v_hidden_size, a_hidden_size,
                 bi_hidden_size, bi_num_attention_heads,
                 attention_probs_dropout_prob, v_attention_probs_dropout_prob,
                 a_attention_probs_dropout_prob,
                 av_attention_probs_dropout_prob,
                 at_attention_probs_dropout_prob, intermediate_size,
                 v_intermediate_size, a_intermediate_size, hidden_act,
                 v_hidden_act, a_hidden_act, hidden_dropout_prob,
                 v_hidden_dropout_prob, a_hidden_dropout_prob):
        super(BertConnectionLayer, self).__init__()
        self.ent_attention = BertEntAttention(
            hidden_size,
            v_hidden_size,
            a_hidden_size,
            bi_hidden_size,
            attention_probs_dropout_prob,
            v_attention_probs_dropout_prob,
            a_attention_probs_dropout_prob,
            av_attention_probs_dropout_prob,
            at_attention_probs_dropout_prob,
            bi_num_attention_heads,
        )

        self.ent_output = BertEntOutput(
            bi_hidden_size,
            hidden_size,
            v_hidden_size,
            v_hidden_dropout_prob,
            hidden_dropout_prob,
        )

        self.v_intermediate = BertIntermediate(v_hidden_size,
                                               v_intermediate_size,
                                               v_hidden_act)
        self.v_output = BertOutput(v_intermediate_size, v_hidden_size,
                                   v_hidden_dropout_prob)

        self.t_intermediate = BertIntermediate(hidden_size, intermediate_size,
                                               hidden_act)
        self.t_output = BertOutput(intermediate_size, hidden_size,
                                   hidden_dropout_prob)

        self.a_intermediate = BertIntermediate(a_hidden_size,
                                               a_intermediate_size,
                                               a_hidden_act)
        self.a_output = BertOutput(a_intermediate_size, a_hidden_size,
                                   a_hidden_dropout_prob)

    def forward(
        self,
        input_tensor1,
        attention_mask1,
        input_tensor2,
        attention_mask2,
        input_tensor3,
        attention_mask3,
    ):

        ent_output1, ent_output2, ent_output3 = self.ent_attention(
            input_tensor1, attention_mask1, input_tensor2, attention_mask2,
            input_tensor3, attention_mask3)

        attention_output1, attention_output2, attention_output3 = self.ent_output(
            ent_output1, input_tensor1, ent_output2, input_tensor2, ent_output3,
            input_tensor3)

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        intermediate_output3 = self.a_intermediate(attention_output3)
        layer_output3 = self.a_output(intermediate_output3, attention_output3)

        return layer_output1, layer_output2, layer_output3


class BertEncoder(nn.Layer):
    """
    ActBert Encoder, consists 3 pathway of multi-BertLayers and BertConnectionLayer.
    """
    def __init__(
        self,
        v_ent_attention_id,
        t_ent_attention_id,
        a_ent_attention_id,
        fixed_t_layer,
        fixed_v_layer,
        hidden_size,
        v_hidden_size,
        a_hidden_size,
        bi_hidden_size,
        intermediate_size,
        v_intermediate_size,
        a_intermediate_size,
        hidden_act,
        v_hidden_act,
        a_hidden_act,
        hidden_dropout_prob,
        v_hidden_dropout_prob,
        a_hidden_dropout_prob,
        attention_probs_dropout_prob,
        v_attention_probs_dropout_prob,
        a_attention_probs_dropout_prob,
        av_attention_probs_dropout_prob,
        at_attention_probs_dropout_prob,
        num_attention_heads,
        v_num_attention_heads,
        a_num_attention_heads,
        bi_num_attention_heads,
        num_hidden_layers,
        v_num_hidden_layers,
        a_num_hidden_layers,
    ):
        super(BertEncoder, self).__init__()
        self.v_ent_attention_id = v_ent_attention_id
        self.t_ent_attention_id = t_ent_attention_id
        self.a_ent_attention_id = a_ent_attention_id
        self.fixed_t_layer = fixed_t_layer
        self.fixed_v_layer = fixed_v_layer

        layer = BertLayer(hidden_size, intermediate_size, hidden_act,
                          hidden_dropout_prob, num_attention_heads,
                          attention_probs_dropout_prob)
        v_layer = BertLayer(v_hidden_size, v_intermediate_size, v_hidden_act,
                            v_hidden_dropout_prob, v_num_attention_heads,
                            v_attention_probs_dropout_prob)
        a_layer = BertLayer(a_hidden_size, a_intermediate_size, a_hidden_act,
                            a_hidden_dropout_prob, a_num_attention_heads,
                            a_attention_probs_dropout_prob)
        connect_layer = BertConnectionLayer(
            hidden_size, v_hidden_size, a_hidden_size, bi_hidden_size,
            bi_num_attention_heads, attention_probs_dropout_prob,
            v_attention_probs_dropout_prob, a_attention_probs_dropout_prob,
            av_attention_probs_dropout_prob, at_attention_probs_dropout_prob,
            intermediate_size, v_intermediate_size, a_intermediate_size,
            hidden_act, v_hidden_act, a_hidden_act, hidden_dropout_prob,
            v_hidden_dropout_prob, a_hidden_dropout_prob)

        self.layer = nn.LayerList(
            [copy.deepcopy(layer) for _ in range(num_hidden_layers)])  #12
        self.v_layer = nn.LayerList(
            [copy.deepcopy(v_layer) for _ in range(v_num_hidden_layers)])  #2
        self.a_layer = nn.LayerList(
            [copy.deepcopy(a_layer) for _ in range(a_num_hidden_layers)])  #3
        self.c_layer = nn.LayerList([
            copy.deepcopy(connect_layer) for _ in range(len(v_ent_attention_id))
        ]  #2  [0,1]
                                    )

    def forward(
        self,
        txt_embedding,
        image_embedding,
        action_embedding,
        txt_attention_mask,
        image_attention_mask,
        action_attention_mask,
        output_all_encoded_layers=True,
    ):
        v_start, a_start, t_start = 0, 0, 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []
        all_encoder_layers_a = []

        for v_layer_id, a_layer_id, t_layer_id in zip(self.v_ent_attention_id,
                                                      self.a_ent_attention_id,
                                                      self.t_ent_attention_id):
            v_end = v_layer_id
            a_end = a_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            ### region embedding
            for idx in range(v_start,
                             self.fixed_v_layer):  #两次训练，这个循环都没有进去  #前面的层固定住
                with paddle.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding, image_attention_mask)
                    v_start = self.fixed_v_layer
            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding, image_attention_mask)

            ### action embedding
            for idx in range(a_start, a_end):
                action_embedding, action_attention_probs = self.a_layer[idx](
                    action_embedding, action_attention_mask)

            ### text embedding
            for idx in range(t_start, self.fixed_t_layer):
                with paddle.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask)
                    t_start = self.fixed_t_layer
            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask)

            image_embedding, txt_embedding, action_embedding = self.c_layer[
                count](image_embedding, image_attention_mask, txt_embedding,
                       txt_attention_mask, action_embedding,
                       action_attention_mask)

            v_start = v_end
            t_start = t_end
            a_start = a_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)
                all_encoder_layers_a.append(action_embedding)

        for idx in range(v_start, len(self.v_layer)):  # 1
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding, image_attention_mask)

        for idx in range(a_start, len(self.a_layer)):
            action_embedding, action_attention_probs = self.a_layer[idx](
                action_embedding, action_attention_mask)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)  #8, 36, 768
            all_encoder_layers_v.append(image_embedding)  #8, 37, 1024
            all_encoder_layers_a.append(action_embedding)  #8, 5, 768

        return all_encoder_layers_t, all_encoder_layers_v, all_encoder_layers_a


class BertPooler(nn.Layer):
    """ "Pool" the model by simply taking the hidden state corresponding
        to the first token.
    """
    def __init__(self, hidden_size, bi_hidden_size):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  #8, 768
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Layer):
    def __init__(
        self,
        vocab_size,
        max_position_embeddings,
        type_vocab_size,
        v_feature_size,
        a_feature_size,
        num_hidden_layers,
        v_num_hidden_layers,
        a_num_hidden_layers,
        v_ent_attention_id,
        t_ent_attention_id,
        a_ent_attention_id,
        fixed_t_layer,
        fixed_v_layer,
        hidden_size,
        v_hidden_size,
        a_hidden_size,
        bi_hidden_size,
        intermediate_size,
        v_intermediate_size,
        a_intermediate_size,
        hidden_act,
        v_hidden_act,
        a_hidden_act,
        hidden_dropout_prob,
        v_hidden_dropout_prob,
        a_hidden_dropout_prob,
        attention_probs_dropout_prob,
        v_attention_probs_dropout_prob,
        a_attention_probs_dropout_prob,
        av_attention_probs_dropout_prob,
        at_attention_probs_dropout_prob,
        num_attention_heads,
        v_num_attention_heads,
        a_num_attention_heads,
        bi_num_attention_heads,
    ):
        super(BertModel, self).__init__()
        # initilize word embedding
        self.embeddings = BertEmbeddings(vocab_size, max_position_embeddings,
                                         type_vocab_size, hidden_size,
                                         hidden_dropout_prob)
        # initlize the region embedding
        self.v_embeddings = BertImageEmbeddings(v_feature_size, v_hidden_size,
                                                v_hidden_dropout_prob)
        # initlize the action embedding
        self.a_embeddings = BertActionEmbeddings(a_feature_size, a_hidden_size,
                                                 a_hidden_dropout_prob)

        self.encoder = BertEncoder(
            v_ent_attention_id, t_ent_attention_id, a_ent_attention_id,
            fixed_t_layer, fixed_v_layer, hidden_size, v_hidden_size,
            a_hidden_size, bi_hidden_size, intermediate_size,
            v_intermediate_size, a_intermediate_size, hidden_act, v_hidden_act,
            a_hidden_act, hidden_dropout_prob, v_hidden_dropout_prob,
            a_hidden_dropout_prob, attention_probs_dropout_prob,
            v_attention_probs_dropout_prob, a_attention_probs_dropout_prob,
            av_attention_probs_dropout_prob, at_attention_probs_dropout_prob,
            num_attention_heads, v_num_attention_heads, a_num_attention_heads,
            bi_num_attention_heads, num_hidden_layers, v_num_hidden_layers,
            a_num_hidden_layers)

        self.t_pooler = BertPooler(hidden_size, bi_hidden_size)
        self.v_pooler = BertPooler(v_hidden_size, bi_hidden_size)
        self.a_pooler = BertPooler(a_hidden_size, bi_hidden_size)

    def forward(
        self,
        text_ids,
        action_feat,
        image_feat,
        image_loc,
        token_type_ids=None,
        text_mask=None,
        image_mask=None,
        action_mask=None,
        output_all_encoded_layers=False,
    ):
        """
        text_ids: input text ids. Shape: [batch_size, seqence_length]
        action_feat: input action feature. Shape: [batch_size, action_length, action_feature_dim]
        image_feat: input image feature. Shape: [batch_size, region_length, image_feature_dim]]
        image_loc: input region location. Shape: [batch_size, region_length, region_location_dim]
        token_type_ids: segment ids of each video clip. Shape: [batch_size, seqence_length]
        text_mask: text mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, seqence_length]
        image_mask: image mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, region_length]
        action_mask: action mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, action_length]
        output_all_encoded_layers: is output encoded layers feature or not. Type: Bool.
        """
        if text_mask is None:
            text_mask = paddle.ones_like(text_ids)
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(text_ids)
        if image_mask is None:
            image_mask = paddle.ones(image_feat.shape[0],
                                     image_feat.shape[1]).astype(text_ids.dtype)
        if action_mask is None:
            action_mask = paddle.ones(action_feat.shape[0],
                                      action_feat.shape[1]).astype(
                                          text_ids.dtype)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length].
        extended_text_mask = text_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_action_mask = action_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        def set_mask(extended_attention_mask):
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

        extended_text_mask = set_mask(extended_text_mask)
        extended_image_mask = set_mask(extended_image_mask)
        extended_action_mask = set_mask(extended_action_mask)

        t_embedding_output = self.embeddings(text_ids, token_type_ids)
        v_embedding_output = self.v_embeddings(image_feat, image_loc)
        a_embedding_output = self.a_embeddings(action_feat)

        # var = [t_embedding_output, v_embedding_output, a_embedding_output]
        # import numpy as np
        # for i, item in enumerate(var):
        #     np.save('tmp/' + str(i)+'.npy', item.numpy())

        encoded_layers_t, encoded_layers_v, encoded_layers_a = self.encoder(
            t_embedding_output,
            v_embedding_output,
            a_embedding_output,
            extended_text_mask,
            extended_image_mask,
            extended_action_mask,
            output_all_encoded_layers=output_all_encoded_layers,
        )

        sequence_output_t = encoded_layers_t[-1]  #get item from list
        sequence_output_v = encoded_layers_v[-1]
        sequence_output_a = encoded_layers_a[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)
        pooled_output_a = self.a_pooler(sequence_output_a)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]
            encoded_layers_a = encoded_layers_a[-1]

        return encoded_layers_t, encoded_layers_v, encoded_layers_a, \
            pooled_output_t, pooled_output_v, pooled_output_a


# For Head
class BertPredictionHeadTransform(nn.Layer):
    def __init__(self, hidden_size, hidden_act):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, str) or (sys.version_info[0] == 2
                                           and isinstance(hidden_act, str)):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Layer):
    def __init__(self, hidden_size, hidden_act, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        assert bert_model_embedding_weights.shape[1] == hidden_size
        vocab_size = bert_model_embedding_weights.shape[0]

        # another implementation which would create another big params:
        # self.decoder = nn.Linear(hidden_size, vocab_size)   # NOTE bias default: constant 0.0
        # self.decoder.weight = self.create_parameter(shape=[hidden_size, vocab_size],
        #                                             default_initializer=nn.initializer.Assign(
        #                                                 bert_model_embedding_weights.t()))  # transpose

        self.decoder_weight = bert_model_embedding_weights
        self.decoder_bias = self.create_parameter(
            shape=[vocab_size],
            dtype=bert_model_embedding_weights.dtype,
            is_bias=True)  # NOTE bias default: constant 0.0

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = paddle.tensor.matmul(
            hidden_states, self.decoder_weight,
            transpose_y=True) + self.decoder_bias
        return hidden_states


class BertImageActionPredictionHead(nn.Layer):
    def __init__(self, hidden_size, hidden_act, target_size):
        super(BertImageActionPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act)

        self.decoder = nn.Linear(hidden_size, target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Layer):
    def __init__(self, hidden_size, v_hidden_size, a_hidden_size,
                 bi_hidden_size, hidden_act, v_hidden_act, a_hidden_act,
                 v_target_size, a_target_size, fusion_method,
                 bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(hidden_size, hidden_act,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(bi_hidden_size, 2)
        self.imagePredictions = BertImageActionPredictionHead(
            v_hidden_size, v_hidden_act, v_target_size)  # visual class number
        self.actionPredictions = BertImageActionPredictionHead(
            a_hidden_size, a_hidden_act, a_target_size)  # action class number
        self.fusion_method = fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(self, sequence_output_t, sequence_output_v, sequence_output_a,
                pooled_output_t, pooled_output_v, pooled_output_a):

        if self.fusion_method == 'sum':
            pooled_output = self.dropout(pooled_output_t + pooled_output_v +
                                         pooled_output_a)
        elif self.fusion_method == 'mul':
            pooled_output = self.dropout(pooled_output_t * pooled_output_v +
                                         pooled_output_a)
        else:
            assert False

        prediction_scores_t = self.predictions(
            sequence_output_t)  # 8， 36 ，30522
        seq_relationship_score = self.seq_relationship(pooled_output)  # 8, 2
        prediction_scores_v = self.imagePredictions(
            sequence_output_v)  # 8, 37, 1601
        prediction_scores_a = self.actionPredictions(
            sequence_output_a)  # 8, 5, 401

        return prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score


@BACKBONES.register()
class BertForMultiModalPreTraining(nn.Layer):
    """BERT model with multi modal pre-training heads.
    """
    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        type_vocab_size=2,
        v_target_size=1601,
        a_target_size=700,
        v_feature_size=2048,
        a_feature_size=2048,
        num_hidden_layers=12,
        v_num_hidden_layers=2,
        a_num_hidden_layers=3,
        t_ent_attention_id=[10, 11],
        v_ent_attention_id=[0, 1],
        a_ent_attention_id=[0, 1],
        fixed_t_layer=0,
        fixed_v_layer=0,
        hidden_size=768,
        v_hidden_size=1024,
        a_hidden_size=768,
        bi_hidden_size=1024,
        intermediate_size=3072,
        v_intermediate_size=1024,
        a_intermediate_size=3072,
        hidden_act="gelu",
        v_hidden_act="gelu",
        a_hidden_act="gelu",
        hidden_dropout_prob=0.1,
        v_hidden_dropout_prob=0.1,
        a_hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        v_attention_probs_dropout_prob=0.1,
        a_attention_probs_dropout_prob=0.1,
        av_attention_probs_dropout_prob=0.1,
        at_attention_probs_dropout_prob=0.1,
        num_attention_heads=12,
        v_num_attention_heads=8,
        a_num_attention_heads=12,
        bi_num_attention_heads=8,
        fusion_method="mul",
        pretrained=None,
    ):
        """
        vocab_size: vocabulary size. Default: 30522.
        max_position_embeddings: max position id. Default: 512.
        type_vocab_size: max segment id. Default: 2.
        v_target_size: class number of visual word. Default: 1601.
        a_target_size: class number of action word. Default: 700.
        v_feature_size: input visual feature dimension. Default: 2048.
        a_feature_size: input action feature dimension. Default: 2048.
        num_hidden_layers: number of BertLayer in text transformer. Default: 12.
        v_num_hidden_layers: number of BertLayer in visual transformer. Default: 2.
        a_num_hidden_layers: number of BertLayer in action transformer. Default:3.
        t_ent_attention_id: index id of BertConnectionLayer in text transformer. Default: [10, 11].
        v_ent_attention_id: index id of BertConnectionLayer in visual transformer. Default:[0, 1].
        a_ent_attention_id: index id of BertConnectionLayer in action transformer. Default:[0, 1].
        fixed_t_layer: index id of fixed BertLayer in text transformer. Default: 0.
        fixed_v_layer: index id of fixed BertLayer in visual transformer. Default: 0.
        hidden_size: hidden size in text BertLayer. Default: 768.
        v_hidden_size: hidden size in visual BertLayer. Default: 1024.
        a_hidden_size: hidden size in action BertLayer. Default: 768.
        bi_hidden_size: hidden size in BertConnectionLayer. Default: 1024,
        intermediate_size: intermediate size in text BertLayer. Default: 3072.
        v_intermediate_size: intermediate size in visual BertLayer. Default: 1024.
        a_intermediate_size: intermediate size in text BertLayer. Default: 3072.
        hidden_act: hidden activation function in text BertLayer. Default: "gelu".
        v_hidden_act: hidden activation function in visual BertLayer. Default: "gelu".
        a_hidden_act: hidden activation function in action BertLayer. Default: "gelu".
        hidden_dropout_prob: hidden dropout probability in text Embedding Layer. Default: 0.1
        v_hidden_dropout_prob: hidden dropout probability in visual Embedding Layer. Default: 0.1
        a_hidden_dropout_prob: hidden dropout probability in action Embedding Layer. Default: 0.1
        attention_probs_dropout_prob: attention dropout probability in text BertLayer. Default: 0.1
        v_attention_probs_dropout_prob: attention dropout probability in visual BertLayer. Default: 0.1
        a_attention_probs_dropout_prob: attention dropout probability in action BertLayer. Default: 0.1
        av_attention_probs_dropout_prob: attention dropout probability in action-visual BertConnectionLayer. Default: 0.1
        at_attention_probs_dropout_prob: attention dropout probability in action-text BertConnectionLayer. Default: 0.1
        num_attention_heads: number of heads in text BertLayer. Default: 12.
        v_num_attention_heads: number of heads in visual BertLayer. Default: 8.
        a_num_attention_heads: number of heads in action BertLayer. Default: 12.
        bi_num_attention_heads: number of heads in BertConnectionLayer. Default: 8.
        fusion_method: methods of fusing pooled output from 3 transformer. Default: "mul".
        """
        super(BertForMultiModalPreTraining, self).__init__()
        self.pretrained = pretrained
        self.vocab_size = vocab_size
        self.a_target_size = a_target_size

        self.bert = BertModel(
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            v_feature_size,
            a_feature_size,
            num_hidden_layers,
            v_num_hidden_layers,
            a_num_hidden_layers,
            v_ent_attention_id,
            t_ent_attention_id,
            a_ent_attention_id,
            fixed_t_layer,
            fixed_v_layer,
            hidden_size,
            v_hidden_size,
            a_hidden_size,
            bi_hidden_size,
            intermediate_size,
            v_intermediate_size,
            a_intermediate_size,
            hidden_act,
            v_hidden_act,
            a_hidden_act,
            hidden_dropout_prob,
            v_hidden_dropout_prob,
            a_hidden_dropout_prob,
            attention_probs_dropout_prob,
            v_attention_probs_dropout_prob,
            a_attention_probs_dropout_prob,
            av_attention_probs_dropout_prob,
            at_attention_probs_dropout_prob,
            num_attention_heads,
            v_num_attention_heads,
            a_num_attention_heads,
            bi_num_attention_heads,
        )
        self.cls = BertPreTrainingHeads(
            hidden_size, v_hidden_size, a_hidden_size, bi_hidden_size,
            hidden_act, v_hidden_act, a_hidden_act, v_target_size,
            a_target_size, fusion_method,
            self.bert.embeddings.word_embeddings.weight)

    def init_weights(self):
        """Initiate the parameters.
        """
        if isinstance(self.pretrained, str) and self.pretrained.strip() != "":
            load_ckpt(self, self.pretrained)
        elif self.pretrained is None or self.pretrained.strip() == "":
            for layer in self.sublayers():
                if isinstance(layer, (nn.Linear, nn.Embedding)):
                    weight_init_(layer, 'Normal', std=0.02)
                elif isinstance(layer, nn.LayerNorm):
                    weight_init_(layer, 'Constant', value=1)

    def forward(
            self,
            text_ids,  #8,36
            action_feat,  #8,5,2048
            image_feat,  #8,37,2048
            image_loc,  #8,37,5
            token_type_ids=None,  #8,36
            text_mask=None,  #8,36
            image_mask=None,  #8,37
            action_mask=None,  #8,5
    ):
        """
        text_ids: input text ids. Shape: [batch_size, seqence_length]
        action_feat: input action feature. Shape: [batch_size, action_length, action_feature_dim]
        image_feat: input image feature. Shape: [batch_size, region_length+1, image_feature_dim]], add 1 for image global feature.
        image_loc: input region location. Shape: [batch_size, region_length+1, region_location_dim], add 1 for image global feature location.
        token_type_ids: segment ids of each video clip. Shape: [batch_size, seqence_length]
        text_mask: text mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, seqence_length]
        image_mask: image mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, region_length]
        action_mask: action mask, 1 for real tokens and 0 for padding tokens. Shape: [batch_size, action_length]
        """
        sequence_output_t, sequence_output_v, sequence_output_a, \
        pooled_output_t, pooled_output_v, pooled_output_a = self.bert(
            text_ids,
            action_feat,
            image_feat,
            image_loc,
            token_type_ids,
            text_mask,
            image_mask,
            action_mask,
            output_all_encoded_layers=False,
        )

        prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score = self.cls(
            sequence_output_t, sequence_output_v, sequence_output_a,
            pooled_output_t, pooled_output_v, pooled_output_a)

        return prediction_scores_t, prediction_scores_v, prediction_scores_a, seq_relationship_score
