#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def download():
    corpus_url="https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request=requests.get(corpus_url)
    corpus=web_request.content
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()

download()


# In[3]:


def load_text8():
    with open("./text8.txt", "r") as f:
        corpus=f.read().strip("\n")
    f.close()

    return corpus

corpus=load_text8()

print(corpus[:500])


# In[4]:


def data_preprocess(corpus):
    corpus=corpus.strip().lower()
    corpus=corpus.split(" ")
    return corpus

corpus=data_preprocess(corpus)
print(corpus[:50])


# In[5]:


def build_dict(corpus):
    word_freq_dict=dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word]=0
        word_freq_dict[word]+=1

    word_freq_dict=sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    word2id_dict=dict()
    word2id_freq=dict()
    id2word_dict=dict()

    for word, freq in word_freq_dict:
        curr_id=len(word2id_dict)
        word2id_dict[word]=curr_id
        word2id_freq[word2id_dict[word]]=freq
        id2word_dict[curr_id]=word

    return word2id_freq, word2id_dict, id2word_dict

word2id_freq, word2id_dict, id2word_dict=build_dict(corpus)
vocab_size=len(word2id_freq)
print("there are totoally %d different words in the corpus"%vocab_size)
for _, (word,word_id) in zip(range(50),word2id_dict.items()):
    print("word %s,its id %d,its word freq %d" % (word,word_id,word2id_freq[word_id]))


# In[6]:


def convert_corpus_to_id(corpus, word2id_dict):
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

corpus = convert_corpus_to_id(corpus, word2id_dict)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])


# In[7]:


def subsampling(corpus, word2id_freq):
    
    def discard(word_id):
        return random.uniform(0, 1)<1-math.sqrt(
            1e-4/word2id_freq[word_id]*len(corpus))

    corpus=[word for word in corpus if not discard(word)]
    return corpus

corpus=subsampling(corpus,word2id_freq)
print("%d tokens in the corpus"%len(corpus))
print(corpus[:50])


# In[ ]:


def build_data(corpus, word2id_dict, word2id_freq, max_window_size = 3, negative_sample_num = 4):
    
    dataset=[]

    for center_word_idx in range(len(corpus)):
        window_size=random.randint(1, max_window_size)
        center_word=corpus[center_word_idx]

        positive_word_range=(max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        positive_word_candidates=[corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1]+1) if idx != center_word_idx]

        for positive_word in positive_word_candidates:
            dataset.append((center_word, positive_word,1))

            i=0
            while i<negative_sample_num:
                negative_word_candidate=random.randint(0, vocab_size-1)

                if negative_word_candidate not in positive_word_candidates:
                    dataset.append((center_word, negative_word_candidate,0))
                    i+=1
    return dataset
corpus_light=corpus[:int(len(corpus)*0.2)]
dataset=build_data(corpus_light, word2id_dict,word2id_freq)
for _,(center_word, target_word, label) in zip(range(50),dataset):
    print("center_word %s, target %s, label %d"%(id2word_dict[center_word],
                                                   id2word_dict[target_word], label))


# In[ ]:


def build_batch(dataset, batch_size, epoch_num):
    
    # center_word_batch缓存batch_size个中心词
    center_word_batch = []
    # target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for center_word, target_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"),                     np.array(target_word_batch).astype("int64"),                     np.array(label_batch).astype("float32")
                center_word_batch = []
                target_word_batch = []
                label_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"),             np.array(target_word_batch).astype("int64"),             np.array(label_batch).astype("float32")

for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)


# In[ ]:


class SkipGram(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding( 
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform( 
                    low=-init_scale, high=init_scale)))

        self.embedding_out = Embedding(
            num_embeddings = self.vocab_size,
            embedding_dim = self.embedding_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(
                    low=-init_scale, high=init_scale)))

    def forward(self, center_words, target_words, label):
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        word_sim = paddle.multiply(center_words_emb, target_words_emb)
        word_sim = paddle.sum(word_sim, axis=-1)
        word_sim = paddle.reshape(word_sim, shape=[-1])
        pred = F.sigmoid(word_sim)

        loss = F.binary_cross_entropy_with_logits(word_sim, label)
        loss = paddle.mean(loss)
        
        return pred, loss


# In[ ]:


batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001

def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))

paddle.set_device('gpu:0')

skip_gram_model = SkipGram(vocab_size, embedding_size)

adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters = skip_gram_model.parameters())

for center_words, target_words, label in build_batch(
    dataset, batch_size, epoch_num):
    center_words_var = paddle.to_tensor(center_words)
    target_words_var = paddle.to_tensor(target_words)
    label_var = paddle.to_tensor(label)
    
    pred, loss = skip_gram_model(
        center_words_var, target_words_var, label_var)

    loss.backward()
    adam.step()
    adam.clear_grad()

    step += 1
    if step % 1000 == 0:
        print("step %d, loss %.3f" % (step, loss.numpy()[0]))

    if step % 10000 ==0:
        get_similar_tokens('movie', 5, skip_gram_model.embedding.weight)
        get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
        get_similar_tokens('chip', 5, skip_gram_model.embedding.weight)


# In[ ]:




