import io
import os
import sys
import math
import random
import requests
import paddle
import numpy as np
import paddle.fluid as fluid
from collections import OrderedDict
import matplotlib.pyplot as plt
from paddle.fluid.dygraph.nn import Embedding

#下载数据用于训练
def download():
    url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request = requests.get(url)
    corpus = web_request.content
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()
download()

#读取text8数据
def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus

#对数据进行预处理，现将字母都转为小写，在用空格进行切词
def preprocess(corpus):
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus

#构造词典，统计每个词的频率，并根据频率赋予idid
def build_dict(corpus):
    word_freq = dict()
    for word in corpus:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1

    word_freq = sorted(word_freq.items(), key = lambda x:x[1], reverse = True)

    #构造3个不同的词典，分别存储，
    #每个词到id的映射关系：word2id_dict
    #每个id出现的频率：word2id_freq
    #每个id到词典映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    #按照频率，从高到低，开始遍历每个单词，并为这个单词构造id
    for word, freq in word_freq:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict

#将语料库中的单词装换为对应的id
def convert_corpus_to_id(corpus, word2id_dict):
    corpus = [word2id_dict[word] for word in corpus]
    return corpus


# 使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


# 构造训练数据
def build_data(corpus, word2id_dict, word2id_freq, max_window_size=3,
               negative_sample_num=4):
    # 使用一个list存储处理好的数据
    dataset = []
    center_word_idx = 0

    # 从左到右，开始枚举每个中心点的位置
    while center_word_idx < len(corpus):
        # 以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        # 当前的中心词就是center_word_idx所指向的词，可以当作正样本
        positive_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词就是上下文
        context_word_range = (
        max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        context_word_candidates = [corpus[idx] for idx in range(context_word_range[0], context_word_range[1] + 1) if
                                   idx != center_word_idx]

        # 对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for context_word in context_word_candidates:
            # 首先把（上下文，正样本，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((context_word, positive_word, 1))

            # 开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate is not positive_word:
                    # 把（上下文，负样本，label=0）的三元组数据放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((context_word, negative_word_candidate, 0))
                    i += 1

        center_word_idx = min(len(corpus) - 1, center_word_idx + window_size)
        if center_word_idx == (len(corpus) - 1):
            center_word_idx += 1
        if center_word_idx % 200000 == 0:
            print(center_word_idx)

    return dataset


# 构造mini-batch，准备对模型进行训练
# 我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    # context_word_batch缓存batch_size个中心词
    context_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []
    # eval_word_batch每次随机生成几个样例，用于在运行阶段对模型做评估，以便更好地可视化训练效果。
    eval_word_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)

        for context_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            context_word_batch.append([context_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 构造训练中评估的样本，这里我们生成'one','king','chip'三个词的同义词，
            # 看模型认为的同义词有哪些
            if len(eval_word_batch) == 0:
                eval_word_batch.append([word2id_dict['one']])
            elif len(eval_word_batch) == 1:
                eval_word_batch.append([word2id_dict['king']])
            elif len(eval_word_batch) == 2:
                eval_word_batch.append([word2id_dict['who']])

            if len(context_word_batch) == batch_size:
                yield epoch, \
                      np.array(context_word_batch).astype("int64"), \
                      np.array(target_word_batch).astype("int64"), \
                      np.array(label_batch).astype("float32"), \
                      np.array(eval_word_batch).astype("int64")
                context_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []

    if len(context_word_batch) > 0:
        yield epoch, \
              np.array(context_word_batch).astype("int64"), \
              np.array(target_word_batch).astype("int64"), \
              np.array(label_batch).astype("float32"), \
              np.array(eval_word_batch).astype("int64")


class CBOW(fluid.dygraph.Layer):
    def __init__(self, name_scope, vocab_size, embedding_size, init_scale=0.1):
        super(CBOW, self).__init__(name_scope)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

        self.embedding_out = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / embedding_size, high=0.5 / embedding_size)))

    def forward(self, context_words, target_words, label, eval_words):
        context_words_emb = self.embedding(context_words)
        target_words_emb = self.embedding_out(target_words)
        eval_words_emb = self.embedding(eval_words)

        word_sim = fluid.layers.elementwise_mul(context_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim=-1)
        pred = fluid.layers.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        word_sim_on_fly = fluid.layers.matmul(eval_words_emb,
                                              self.embedding._w, transpose_y=True)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss, word_sim_on_fly

def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))

corpus = load_text8()
corpus = preprocess(corpus)
word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)
corpus = convert_corpus_to_id(corpus, word2id_dict)
corpus = subsampling(corpus, word2id_freq)
dataset = build_data(corpus, word2id_dict, word2id_freq)

#开始训练，定义一些训练过程中需要使用的超参数
batch_size = 512
epoch_num = 1
embedding_size = 200
step = 0
learning_rate = 0.001
LOSS = []

with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    cbow_model = CBOW("cbow_model", vocab_size, embedding_size)
    #构造训练这个网络的优化器
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)

    #使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for epoch_num, context_words, target_words, label, eval_words in build_batch(
        dataset, batch_size, epoch_num):
        # print(eval_words.shape[0])
        #使用fluid.dygraph.to_variable函数，将一个numpy的tensor，转换为飞桨可计算的tensor
        context_words_var = fluid.dygraph.to_variable(context_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)
        eval_words_var = fluid.dygraph.to_variable(eval_words)
        
        #将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
        pred, loss, word_sim_on_fly = cbow_model(
            context_words_var, target_words_var, label_var, eval_words_var)

        #通过backward函数，让程序自动完成反向计算
        loss.backward()
        #通过minimize函数，让程序根据loss，完成一步对参数的优化更新
        adam.minimize(loss)
        #使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
        cbow_model.clear_gradients()

        #每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        step += 1
        if step % 1000 == 0:
            LOSS.append(loss.numpy()[0])
            print("epoch num:%d, step %d, loss %.3f" % (epoch_num, step, loss.numpy()[0]))

    get_similar_tokens('one', 5, cbow_model.embedding._w)
    get_similar_tokens('who', 5, cbow_model.embedding._w)
    get_similar_tokens('king', 5, cbow_model.embedding._w)

    plt.plot(range(len(LOSS)), LOSS)
