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


import os
import re
import tarfile
import random
import requests
import numpy as np


# 下载语料用来训练word2vec
def download(save_path, corpus_url="https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"):
    # 通过python的requests类，下载数据集
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    # 将下载的文件写在当前目录的aclImdb_v1.tar.gz文件内
    with open(save_path, "wb") as f:
        f.write(corpus)


# 加载数据至内存，并进行简单处理
def load_imdb(path, is_training):
    # 校验传入路径
    if not os.path.exists(path):
        raise Exception(f"file not exist: {path}")

    data_set = []

    # aclImdb_v1.tar.gz解压后是一个目录
    # 我们可以使用python的tarfile库进行解压
    # 训练数据和测试数据已经经过切分，其中训练数据的地址为：
    # ./aclImdb/train/pos/ 和 ./aclImdb/train/neg/，分别存储着正向情感的数据和负向情感的数据
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感
    for label in ["pos", "neg"]:
        with tarfile.open(path) as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)
            tf = tarf.next()
            while tf is not None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()

    return data_set


# 对数据做预处理
def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")

        data_set.append((sentence, sentence_label))

    return data_set


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
        if not drop_last and len(sentence_batch) > 0:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


# 加载词典
def load_dict(path):
    word2id_dict = {}
    with open(path, "r") as f:
        for line in f:
            word, word_id = line.rstrip().rsplit("\t", maxsplit=1)
            word2id_dict[word] = int(word_id)

    return word2id_dict