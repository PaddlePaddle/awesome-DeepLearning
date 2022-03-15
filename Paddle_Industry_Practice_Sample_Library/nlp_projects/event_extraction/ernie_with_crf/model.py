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


class EventExtractionModel(paddle.nn.Layer):
    def __init__(self, ernie, num_classes=2, dropout=None):
        super(EventExtractionModel, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie
        self.dropout = nn.Dropout(dropout if dropout is not None else self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_classes+2) # add start and stop tag
        
        self.crf = crf.LinearChainCrf(num_classes, crf_lr=0.001, with_start_stop_tag=True)
        self.crf_loss = crf.LinearChainCrfLoss(self.crf)
        self.viterbi_decoder = crf.ViterbiDecoder(self.crf.transitions)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, _ = self.ernie(input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        return emissions
     
    def get_crf_loss(self, emissions, lens, tags):
        loss = self.crf_loss(emissions, lens, tags)
        loss = paddle.mean(loss)
        return loss
 

