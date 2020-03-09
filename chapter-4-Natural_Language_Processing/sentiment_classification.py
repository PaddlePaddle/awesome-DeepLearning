#encoding=utf8
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import re
import sys
import six
import requests
import string
import tarfile
import hashlib
from collections import OrderedDict 
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.nn import Embedding

def download():
    #通过python的requests类，下载存储在
    #https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz的文件
    corpus_url = "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    #将下载的文件写在当前目录的aclImdb_v1.tar.gz文件内
    with open("./aclImdb_v1.tar.gz", "wb") as f:
        f.write(corpus)
    f.close()

download()


def load_imdb(is_training):
    data_set = []

    #aclImdb_v1.tar.gz解压后是一个目录
    #我们可以使用python的rarfile库进行解压
    #训练数据和测试数据已经经过切分，其中训练数据的地址为：
    #./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    #我们把数据依次读取出来，并放到data_set里
    #data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感
    
    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label)) 
                tf = tarf.next()

    return data_set

train_corpus = load_imdb(True)
test_corpus = load_imdb(False)

for i in range(5):
    print("sentence %d, %s" % (i, train_corpus[i][0]))    
    print("sentence %d, label %d" % (i, train_corpus[i][1]))


def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        #这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        #一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")
        
        data_set.append((sentence, sentence_label))

    return data_set

train_corpus = data_preprocess(train_corpus)
test_corpus = data_preprocess(test_corpus)
print(train_corpus[:5])


#构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    word2id_dict = dict()
    word2id_freq = dict()

    #一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict

word2id_freq, word2id_dict = build_dict(train_corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))

#把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        #将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        #这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        #如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict \
                    else word2id_dict['[oov]'] for word in sentence]    
        data_set.append((sentence, sentence_label))
    return data_set

train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
print("%d tokens in the corpus" % len(train_corpus))
print(train_corpus[:5])
print(test_corpus[:5])

#编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle = True):

    #模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，
    #    每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num): 

        #每个epcoh前都shuffle一下数据，有助于提高模型训练的效果
        #但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])
            
            #飞桨在1.6.1要求输入数据必须是形状为[batch_size, max_seq_len，1]的张量
            #在飞桨的后续版本中不再存在类似的要求
            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []

    if len(sentence_batch) == batch_size:
        yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")

for _, batch in zip(range(10), build_batch(word2id_dict, 
                    train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
    print(batch)


#使用飞桨实现一个长短时记忆模型
class SimpleLSTMRNN(fluid.Layer):
    
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None):
        
        #这个模型有几个参数：
        #1. hidden_size，表示embedding-size，或者是记忆向量的维度
        #2. num_steps，表示这个长短时记忆网络，最多可以考虑多长的时间序列
        #3. num_layers，表示这个长短时记忆网络内部有多少层，我们知道，
        #   给定一个形状为[batch_size, seq_len, embedding_size]的输入，
        # 长短时记忆网络会输出一个同样为[batch_size, seq_len, embedding_size]的输出，
        #   我们可以把这个输出再链到一个新的长短时记忆网络上
        # 如此叠加多层长短时记忆网络，有助于学习更复杂的句子甚至是篇章。
        #5. init_scale，表示网络内部的参数的初始化范围，
        # 长短时记忆网络内部用了很多tanh，sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果，
        
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        # weight_1_arr用于存储不同层的长短时记忆网络中，不同门的W参数
        self.weight_1_arr = []
        self.weight_2_arr = []
        # bias_arr用于存储不同层的长短时记忆网络中，不同门的b参数
        self.bias_arr = []
        self.mask_array = []

        # 通过使用create_parameter函数，创建不同长短时记忆网络层中的参数
        # 通过上面的公式，我们知道，我们总共需要8个形状为[_hidden_size, _hidden_size]的W向量
        # 和4个形状为[_hidden_size]的b向量，因此，我们在声明参数的时候，
        # 一次性声明一个大小为[self._hidden_size * 2, self._hidden_size * 4]的参数
        # 和一个 大小为[self._hidden_size * 4]的参数，这样做的好处是，
        # 可以使用一次矩阵计算，同时计算8个不同的矩阵乘法
        # 以便加快计算速度
        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    # 定义LSTM网络的前向计算逻辑，飞桨会自动根据前向计算结果，给出反向结果
    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []
        
        #输入有三个信号：
        # 1. input_embedding，这个就是输入句子的embedding表示，
        # 是一个形状为[batch_size, seq_len, embedding_size]的张量
        # 2. init_hidden，这个表示LSTM中每一层的初始h的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空
        # 3. init_cell，这个表示LSTM中每一层的初始c的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空

        # 我们需要通过slice操作，把每一层的初始hidden和cell值拿出来，
        # 并存储在cell_array和hidden_array中
        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        # res记录了LSTM中每一层的输出结果（hidden）
        res = []
        for index in range(self._num_steps):
            # 首先需要通过slice函数，拿到输入tensor input_embedding中当前位置的词的向量表示
            # 并把这个词的向量表示转换为一个大小为 [batch_size, embedding_size]的张量
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])
            
            # 计算每一层的结果，从下而上
            for k in range(self._num_layers):
                # 首先获取每一层LSTM对应上一个时间步的hidden，cell，以及当前层的W和b参数
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                # 我们把hidden和拿到的当前步的input拼接在一起，便于后续计算
                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                
                # 将输入门，遗忘门，输出门等对应的W参数，和输入input和pre-hidden相乘
                # 我们通过一步计算，就同时完成了8个不同的矩阵运算，提高了运算效率
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                # 将b参数也加入到前面的运算结果中
                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                
                # 通过split函数，将每个门得到的结果拿出来
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                
                # 把输入门，遗忘门，输出门等对应的权重作用在当前输入input和pre-hidden上
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                
                # 记录当前步骤的计算结果，
                # m是当前步骤需要输出的hidden
                # c是当前步骤需要输出的cell
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m
                
                # 一般来说，我们有时候会在LSTM的结果结果内加入dropout操作
                # 这样会提高模型的训练鲁棒性
                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
                    
            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))
        
        # 计算长短时记忆网络的结果返回回来，包括：
        # 1. real_res：每个时间步上不同层的hidden结果
        # 2. last_hidden：最后一个时间步中，每一层的hidden的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        # 3. last_cell：最后一个时间步中，每一层的cell的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        
        return real_res, last_hidden, last_cell


# 定义一个可以用于情感分类的网络
class SentimentClassifier(fluid.Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 class_num=2,
                 num_layers=1,
                 num_steps=128,
                 init_scale=0.1,
                 dropout=None):
        
        #这个模型的参数分别为：
        #1. hidden_size，表示embedding-size，hidden已经cell向量的维度
        #2. vocab_size，模型可以考虑的词表大小
        #3. class_num，情感类型个数，可以是2分类，也可以是多分类
        #4. num_steps，表示这个情感分析模型最大可以考虑的句子长度
        #5. init_scale，表示网络内部的参数的初始化范围，
        # 长短时记忆网络内部用了很多tanh，sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果
        
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        # 声明一个LSTM模型，用来把一个句子抽象城一个向量
        self.simple_lstm_rnn = SimpleLSTMRNN(
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        
        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = Embedding(
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        
        # 在得到一个句子的向量表示后，我们需要根据这个向量表示对这个句子进行分类
        # 一般来说，我们可以把这个句子的向量表示，
        # 乘以一个大小为[self.hidden_size, self.class_num]的W参数
        # 并加上一个大小为[self.class_num]的b参数
        # 通过这种手段达到把句子向量映射到分类结果的目标
        
        # 我们需要声明最终在使用句子向量映射到具体情感类别过程中所需要使用的参数
        # 这个参数的大小一般是[self.hidden_size, self.class_num]
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        # 同样的，我们需要声明最终分类过程中的b参数
        #  这个参数的大小一般是[self.class_num]
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.class_num],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label):

        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')
        init_cell_data = np.zeros(
            (1, batch_size, embedding_size), dtype='float32')

        # 将这些初始记忆转换为飞桨可计算的向量
        # 并设置stop-gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = fluid.dygraph.to_variable(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = fluid.dygraph.to_variable(init_cell_data)
        init_cell.stop_gradient = True

        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        # 将输入的句子的mini-batch input，转换为词向量表示
        x_emb = self.embedding(input)

        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.dropout,
                dropout_implementation='upscale_in_train')
        
        # 使用LSTM网络，把每个句子转换为向量表示
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self.hidden_size])
        
        # 将每个句子的向量表示，通过矩阵计算，映射到具体的情感类别上
        projection = fluid.layers.matmul(last_hidden, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.class_num])
        pred = fluid.layers.softmax(projection, axis=-1)
        
        # 根据给定的标签信息，计算整个网络的损失函数，这里我们可以直接使用分类任务中常使用的交叉熵来训练网络
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reduce_mean(loss)

        # 最终返回预测结果pred，和网络的loss
        return pred, loss

import paddle.fluid as fluid
#开始训练
batch_size = 128
epoch_num = 5
embedding_size = 256
step = 0
learning_rate = 0.01
max_seq_len = 128

with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    # 创建一个用于情感分类的网络实例，sentiment_classifier
    sentiment_classifier = SentimentClassifier(
    	embedding_size, vocab_size, num_steps=max_seq_len)
    # 创建优化器AdamOptimizer，用于更新这个网络的参数
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate, parameter_list = sentiment_classifier.parameters())

    for sentences, labels in build_batch(
        word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len):
        
        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        loss.backward()
        adam.minimize(loss)
        sentiment_classifier.clear_gradients()
        
        step += 1
        if step % 10 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))
            
    # 我们希望在网络训练结束以后评估一下训练好的网络的效果
    # 通过eval()函数，将网络设置为eval模式，在eval模式中，网络不会进行梯度更新
    sentiment_classifier.eval()
    # 这里我们需要记录模型预测结果的准确率
    # 对于二分类任务来说，准确率的计算公式为：
    # (true_positive + true_negative) / 
    # (true_positive + true_negative + false_positive + false_negative)
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for sentences, labels in build_batch(
        word2id_dict, test_corpus, batch_size, 1, max_seq_len):
        
        sentences_var = fluid.dygraph.to_variable(sentences)
        labels_var = fluid.dygraph.to_variable(labels)
        
        # 获取模型对当前batch的输出结果
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()
        for i in range(len(pred)):
            if labels[i][0] == 1:
                if pred[i][1] > pred[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i][1] > pred[i][0]:
                    fp += 1
                else:
                    tn += 1

    # 输出最终评估的模型效果
    print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))























