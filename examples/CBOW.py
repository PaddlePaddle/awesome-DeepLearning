import io
import os
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F


def download():
    # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    # 使用python的requests包下载数据集到本地
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    # 把下载后的文件存储在当前目录的text8.txt文件内
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()


download()

def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus

corpus = load_text8()


def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus


corpus = data_preprocess(corpus)
corpus[:50]


def build_dict(corpus):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    # 构造3个不同的词典，分别存储，
    # 每个词到id的映射关系：word2id_dict
    # 每个id出现的频率：word2id_freq
    # 每个id到词的映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    # 按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)

#print("there are totoally %d different words in the corpus" % vocab_size)
#for _, (word, word_id) in zip(range(10), word2id_dict.items()):
    #print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


# In[57]:


def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus


corpus = convert_corpus_to_id(corpus, word2id_dict)


def subsampling(corpus, word2id_freq):
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus


corpus = subsampling(corpus, word2id_freq)
#print("%d tokens in the corpus" % len(corpus))
#print(corpus[:20])


def build_data(corpus, word2id_dict, word2id_freq, max_window_size=3, negative_sample_num=4):
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

    return dataset


corpus_light = corpus[:int(len(corpus) * 0.2)]
dataset = build_data(corpus_light, word2id_dict, word2id_freq)

#for _, (center_word, target_word, label) in zip(range(25), dataset):
# print("center_word %s, target %s, label %d" % (id2word_dict[center_word],
                                                   #id2word_dict[target_word], label))


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

            # 构造训练中评估的样本，这里我们生成'one','king','who'三个词的同义词，
            # 看模型认为的同义词有哪些
            if len(eval_word_batch) == 0:
                eval_word_batch.append([word2id_dict['one']])
            elif len(eval_word_batch) == 1:
                eval_word_batch.append([word2id_dict['king']])
            elif len(eval_word_batch) == 2:
                eval_word_batch.append([word2id_dict['who']])

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(context_word_batch) == batch_size:
                yield epoch, np.array(context_word_batch).astype("int64"), np.array(target_word_batch).astype(
                    "int64"), np.array(label_batch).astype("float32"), np.array(eval_word_batch).astype("int64")
                context_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []

    if len(context_word_batch) > 0:
        yield epoch, np.array(context_word_batch).astype("int64"), np.array(target_word_batch).astype(
            "int64"), np.array(label_batch).astype("float32"), np.array(eval_word_batch).astype("int64")


for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)


class CBOW(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个skipgram这个模型的词表大小
        # embedding_size定义了词向量的维度是多少
        # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(CBOW, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # 使用Embedding函数构造一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 数据类型为：float32
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))

        # 使用Embedding函数构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding_out = Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))

    # 定义网络的前向计算逻辑
    # center_words是一个tensor（mini-batch），表示中心词
    # target_words是一个tensor（mini-batch），表示目标词
    # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, context_words, target_words, label, eval_words):
        # 首先，通过self.embedding参数，将mini-batch中的词转换为词向量
        # 这里center_words和eval_words_emb查询的是一个相同的参数
        # 而target_words_emb查询的是另一个参数
        context_words_emb = self.embedding(context_words)
        target_words_emb = self.embedding_out(target_words)
        eval_words_emb = self.embedding(eval_words)

        # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = paddle.multiply(context_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数，注意我们使用的是binary_cross_entropy_with_logits函数
        # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)

        # 我们通过一个矩阵乘法，来对每个词计算他的同义词
        # on_fly在机器学习或深度学习中往往指在在线计算中做什么，
        # 比如我们需要在训练中做评估，就可以说evaluation_on_fly
        # word_sim_on_fly = paddle.matmul(eval_words_emb,
        #     self.embedding._w, transpose_y = True)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss  # , word_sim_on_fly


batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001


# 定义一个使用word-embedding查询同义词的函数
# 这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
# 我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
# 具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
# 两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))


# 将模型放到GPU上训练
paddle.set_device('gpu:0')

# 通过我们定义的CBOW类，来构造一个Skip-gram模型网络
cbow_model = CBOW(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=cbow_model.parameters())

# 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
for epoch_num, context_words, target_words, label, eval_words in build_batch(
        dataset, batch_size, epoch_num):
    # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
    context_words_var = paddle.to_tensor(context_words)
    target_words_var = paddle.to_tensor(target_words)
    label_var = paddle.to_tensor(label)
    eval_words_var = paddle.to_tensor(eval_words)

    # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
    pred, loss = cbow_model(
        context_words_var, target_words_var, label_var, eval_words_var)

    # 程序自动完成反向计算
    loss.backward()
    # 程序根据loss，完成一步对参数的优化更新
    adam.step()
    # 清空模型中的梯度，以便于下一个mini-batch进行更新
    adam.clear_grad()

    # 每经过1000个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
    step += 1
    if step % 1000 == 0:
        print("step %d, loss %.3f" % (step, loss.numpy()[0]))

    # 每隔10000步，打印一次模型对以下查询词的相似词，这里我们使用词和词之间的向量点积作为衡量相似度的方法，只打印了5个最相似的词
    if step % 10000 == 0:
        get_similar_tokens('movie', 5, cbow_model.embedding.weight)
        get_similar_tokens('one', 5, cbow_model.embedding.weight)
        get_similar_tokens('who', 5, cbow_model.embedding.weight)
