#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[ ]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/work')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[ ]:


# encoding=utf8
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


# In[ ]:


# 读取text8数据
def load_text8():
    with open("data/data98805/text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus

corpus = load_text8()

# 打印前500个字符，简要看一下这个语料的样子
print(corpus[:500])


# In[ ]:


# 对语料进行预处理（分词）
def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus

corpus = data_preprocess(corpus)
print(corpus[:50])


# In[ ]:


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    # 首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    # 一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
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
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


# In[ ]:


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

corpus = convert_corpus_to_id(corpus, word2id_dict)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])


# In[ ]:


# 使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    
    # 这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    # 如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus

corpus = subsampling(corpus, word2id_freq)
print("%d tokens in the corpus" % len(corpus))
print(len(corpus))
print(corpus[:50])


# In[ ]:


# 构造数据，准备模型训练
# window_size代表了窗口的大小，一个中心词对应着2×window_size个上下文词
# negative_sample_num代表了每2×window_size个上下文词对应的非中心词个数
def build_data(corpus, word2id_dict, word2id_freq, window_size = 2, negative_sample_num = 4):
    
    # 使用一个list存储处理好的数据
    dataset = []

    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(window_size,len(corpus)-window_size-1):
        # 当前的中心词就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]
        # 以当前中心词为中心，左右两侧在window_size内的词都为上下文词
        context = [corpus[center_word_idx-2], corpus[center_word_idx-1], corpus[center_word_idx+1], corpus[center_word_idx+2]]
        # 把（上下文词，中心词，label=1）的三元组数据放入dataset中，
        dataset.append((context, center_word, 1))

        # 开始负采样
        i = 0
        while i < negative_sample_num:
            non_center_word = random.randint(0, vocab_size-1)
            if non_center_word != center_word:
                # 把（上下文词，非中心词，label=0）的三元组数据放入dataset中
                dataset.append((context, non_center_word, 0))
                i += 1

    return dataset
corpus_light = corpus[:int(len(corpus)*0.2)]
dataset = build_data(corpus_light, word2id_dict, word2id_freq)
for _, (context, result, label) in zip(range(50), dataset):
    print("positive_word %s, result %s, label %d" % ([id2word_dict[i] for i in context],
                                                   id2word_dict[result], label))


# In[ ]:


# 构造mini-batch，准备对模型进行训练
# 我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    
    # word_batch缓存batch_size个上下文词样本，每个样本包含四个上下文词
    word_batch = []
    # result_batch缓存batch_size个结果（可以是中心词或者非中心词）
    result_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for word, result, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            word_batch.append(word)
            result_batch.append([result])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(word_batch) == batch_size:
                yield np.array(word_batch).astype("int64"),                     np.array(result_batch).astype("int64"),                     np.array(label_batch).astype("float32")
                word_batch = []
                result_batch = []
                label_batch = []

    if len(word_batch) > 0:
        yield np.array(word_batch).astype("int64"),             np.array(result_batch).astype("int64"),             np.array(label_batch).astype("float32")

for _, batch in zip(range(1), build_batch(dataset, 128, 3)):
    print(batch)


# In[ ]:


#定义CBOW训练网络结构
#使用paddlepaddle的2.0.2版本
#一般来说，在使用paddle训练的时候，我们需要通过一个类来定义网络结构，这个类继承了paddle.nn.layer
class CBOW(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个CBOW这个模型的词表大小
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
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform( 
                    low=-init_scale, high=init_scale)))

        # 使用Embedding函数构造另外一个词向量参数
        # 这个参数的大小为：[self.vocab_size, self.embedding_size]
        # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding_out = Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))

    # 定义网络的前向计算逻辑
    # context是一个tensor（mini-batch），表示上下文词
    # result是一个tensor（mini-batch），表示最终结果，即是否为中心词
    # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果


    def forward(self, word, result, label):
        # 首先，通过self.embedding参数，将mini-batch中的上下文词转换为词向量
        # 接着将得到的上下文词向量相加得到上下文的隐含表示
        emb = self.embedding(word)/4
        emb = paddle.sum(emb,axis=1)
        word_emb=paddle.fluid.layers.reshape(emb,(-1,1,200))

        result_emb = self.embedding_out(result)

        # 我们通过点乘的方式计算上下文词到中心词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
        word_sim = paddle.multiply(word_emb, result_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        # 通过估计的输出概率定义损失函数，注意我们使用的是binary_cross_entropy_with_logits函数
        # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss


# In[18]:


# 开始训练，定义一些训练过程中需要使用的超参数
batch_size = 128
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001

#注：该函数只是名字没有改，内部都改了
#定义一个使用上下词查询中心词的函数
#这个函数words上下文词，embed是我们学习到的word-embedding参数，vocab_size是词表大小
#我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
#具体实现如下，x代表上下文词相加并求平均后的Embedding，Embedding参数矩阵W代表所有词的Embedding
#两者计算Cos得出所有词对查询词的相似度得分向量，最终取最高的那个词作为中心词

def get_similar_tokens(words, embed, vocab_size):
    W = embed.numpy()
    m=len(words)
    x=[]
    for i in range(m):
        x.append(W[word2id_dict[words[i]]])
    x=np.array(x).astype('float32')
    x=np.sum(x,axis=0)

    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -5)[-5:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        if id2word_dict[i] not in words:
            print('for context %s, the center word is %s' % (words, str(id2word_dict[i])))
            break
        


# 将模型放到GPU上训练
paddle.set_device('gpu:0')

# 通过我们定义的SkipGram类，来构造一个Skip-gram模型网络
CBOW_model = CBOW(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters = CBOW_model.parameters())

# 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
for word, result, label in build_batch(
    dataset, batch_size, epoch_num):
    # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
    word_var = paddle.to_tensor(word)
    result_var = paddle.to_tensor(result)
    label_var = paddle.to_tensor(label)
    
    # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
    pred, loss = CBOW_model(
        word_var, result_var, label_var)

    # 程序自动完成反向计算
    loss.backward()
    # 程序根据loss，完成一步对参数的优化更新
    adam.step()
    # 清空模型中的梯度，以便于下一个mini-batch进行更新
    adam.clear_grad()

    # 每经3000个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
    if step % 3000== 0:
        print("step %d, loss %.3f" % (step, loss.numpy()[0]))

    # 每隔30000步，打印一次模型对以下上下词的对应的中心词，这里我们使用词和词之间的向量点积作为衡量相似度的方法
    
    if step % 30000 ==0:
        get_similar_tokens(['four','two','five','nine'], CBOW_model.embedding.weight, vocab_size)
        get_similar_tokens(['economist','son','alfonso','throne'], CBOW_model.embedding.weight, vocab_size)
    

    step += 1

