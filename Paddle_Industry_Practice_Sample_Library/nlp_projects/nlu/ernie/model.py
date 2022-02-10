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
from paddle import nn
import paddle.nn.functional as F
from paddlenlp.transformers import ErniePretrainedModel

class JointModel(paddle.nn.Layer):
    def __init__(self, ernie, num_slots, num_intents, use_history=False, dropout=None):
        super(JointModel, self).__init__()
        self.num_slots = num_slots
        self.num_intents = num_intents
        self.use_history = use_history

        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])

        if self.use_history:
            self.intent_hidden = nn.Linear(2 * self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
            self.slot_hidden = nn.Linear(2 * self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
        else:
            self.intent_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])
            self.slot_hidden = nn.Linear(self.ernie.config["hidden_size"], self.ernie.config["hidden_size"])

        self.intent_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_intents)
        self.slot_classifier = nn.Linear(self.ernie.config["hidden_size"], self.num_slots)


    def forward(self, token_ids, token_type_ids=None, position_ids=None, attention_mask=None, history_ids=None):
        sequence_output, pooled_output = self.ernie(token_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)

        if self.use_history and (history_ids is not None):
            history_pooled_output = self.ernie(history_ids)[1]
            # concat sequence_output and history output
            sequence_output = paddle.concat([history_pooled_output.unsqueeze(1).tile(repeat_times=[1, sequence_output.shape[1], 1]), sequence_output], axis=-1)
            pooled_output = paddle.concat([history_pooled_output, pooled_output], axis=-1)

        sequence_output = F.relu(self.slot_hidden(self.dropout(sequence_output)))
        pooled_output = F.relu(self.intent_hidden(self.dropout(pooled_output)))

        intent_logits = self.intent_classifier(pooled_output)

        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits

