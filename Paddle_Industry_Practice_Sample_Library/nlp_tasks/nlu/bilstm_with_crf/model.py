# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


import paddle
import paddle.nn as nn
import paddle_crf as crf
import paddle.nn.functional as F

class JointModel(paddle.nn.Layer):
    
    def __init__(self, vocab_size, embedding_size, hidden_size, num_intents, num_slots,  num_layers=1, drop_p=0.1):
        super(JointModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop_p = drop_p
        self.num_intents = num_intents
        self.num_slots = num_slots

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(p=drop_p)
        self.layer_norm = nn.LayerNorm(2*hidden_size)
        self.bilstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, direction="bidirectional", num_layers=num_layers, dropout=drop_p)
        self.ner_classifier = nn.Linear(hidden_size*2, num_slots+2)
        self.intent_classifier = nn.Linear(hidden_size*2, num_intents)

        self.crf = crf.LinearChainCrf(num_slots, crf_lr=0.001, with_start_stop_tag=True)
        self.crf_loss = crf.LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = crf.ViterbiDecoder(self.crf.transitions)


    def forward(self, inputs, lens):
        batch_size, seq_len = inputs.shape
        inputs_embedding = self.embedding(inputs)
        if self.drop_p:
             inputs_embedding = self.dropout(inputs_embedding)
        lstm_outputs, _ = self.bilstm(inputs_embedding)
        lstm_outputs = self.layer_norm(lstm_outputs)
        emissions = self.ner_classifier(lstm_outputs)
        indices = paddle.stack([paddle.arange(batch_size), lens-1], axis=1)
        last_step_hiddens = paddle.gather_nd(lstm_outputs, indices)
        intent_logits = self.intent_classifier(last_step_hiddens)

        return emissions, intent_logits


    def get_slot_loss(self, features, lens, tags):
        slot_loss = self.crf_loss(features, lens, tags)
        slot_loss = paddle.mean(slot_loss)
        return slot_loss

    
    def get_intent_loss(self, intent_logits, intent_labels):
        return F.cross_entropy(intent_logits, intent_labels)





