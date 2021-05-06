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


import paddle
import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
import numpy as np

class SentimentClassifier(paddle.nn.Layer):
    def __init__(self, hidden_size, vocab_size, class_num=2, num_steps=128, num_layers=1, init_scale=0.1, dropout=None):

        # 参数含义如下：
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.class_num，情感类型个数，可以是2分类，也可以是多分类
        # 4.num_steps，表示这个情感分析模型最大可以考虑的句子长度
        # 5.num_layers，表示网络的层数
        # 6.init_scale，表示网络内部的参数的初始化范围
        # 长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果

        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        # 声明一个LSTM模型，用来把每个句子抽象成向量
        self.simple_lstm_rnn = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)

        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, sparse=False,
                                   weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))

        # 在得到一个句子的向量表示后，需要根据这个向量表示对这个句子进行分类
        # 一般来说，可以把这个句子的向量表示乘以一个大小为[self.hidden_size, self.class_num]的W参数，
        # 并加上一个大小为[self.class_num]的b参数，从而达到把句子向量映射到分类结果的目的

        # 我们需要声明最终在使用句子向量映射到具体情感类别过程中所需要使用的参数
        # 这个参数的大小一般是[self.hidden_size, self.class_num]
        self.cls_fc = Linear(in_features=self.hidden_size, out_features=self.class_num,
                             weight_attr=None, bias_attr=None)
        self.dropout_layer = Dropout(p=self.dropout, mode='upscale_in_train')

    def forward(self, input, label):
        batch_size = len(input)

        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')

        # 将这些初始记忆转换为飞桨可计算的向量
        # 设置stop_gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = paddle.to_tensor(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = paddle.to_tensor(init_cell_data)
        init_cell.stop_gradient = True

        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])
        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        # 将输入的句子的mini-batch转换为词向量表示
        x_emb = self.embedding(input)
        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = self.dropout_layer(x_emb)

        # 使用LSTM网络，把每个句子转换为向量表示
        rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb, (init_h, init_c))
        last_hidden = paddle.reshape(
            last_hidden[-1], shape=[-1, self.hidden_size])

        # 将每个句子的向量表示映射到具体的情感类别上
        projection = self.cls_fc(last_hidden)
        pred = F.softmax(projection, axis=-1)

        # 根据给定的标签信息，计算整个网络的损失函数，这里我们可以直接使用分类任务中常使用的交叉熵来训练网络
        loss = F.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = paddle.mean(loss)

        # 最终返回预测结果pred，和网络的loss
        return pred, loss
