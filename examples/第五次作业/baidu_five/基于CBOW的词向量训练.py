#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[3]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
# !mkdir /home/aistudio/external-libraries
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries


# In[4]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
# import sys 
# sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[5]:


import os
import io
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F
import paddle.nn as nn


# In[6]:


# def download():
#     corpus_url='https://dataset.bj.bcebos.com/work2vec/text8.txt'
#     web_request=requests.get(corpus_url)
#     print(web_request)
#     corpus=web_request.content
#     with open("./text8.txt","wb") as f:
#         f.write(corpus)
#     f.close()
# download()


# **加载数据集**

# In[7]:


def load_text8():
    with open("./data/data98805/text8.txt",'r') as f:
        corpus=f.read().strip("\n")
    f.close()
    return corpus
corpus=load_text8()
print(corpus[:100])


# **转换小写，切分单词**

# In[8]:


def data_preprocess(corpus):
    corpus=corpus.strip().lower()
    corpus=corpus.split(" ")
    return corpus
    
corpus=data_preprocess(corpus)
print(len(corpus))
print(corpus[:20])


# In[9]:


corpus=corpus[:1000000]
print(len(corpus))


# **构造词典，内含单词、编号、频次**

# In[10]:


def build_dict(corpus):
    word_freq_dict=dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word]=0
        word_freq_dict[word]+=1
    
    word_freq_dict=sorted(word_freq_dict.items(),key=lambda x:x[1],reverse=True)

    word2id_dict=dict()
    word2id_freq=dict()
    id2word_dict=dict()

    for word,freq in word_freq_dict:
        curr_id=len(word2id_dict)
        word2id_dict[word]=curr_id
        word2id_freq[curr_id]=freq
        id2word_dict[curr_id]=word
    return word2id_dict,word2id_freq,id2word_dict

word2id_dict,word2id_freq,id2word_dict=build_dict(corpus)
vocab_size=len(word2id_freq)
print('there are totally {} different words in the corpus'.format(vocab_size))
for _,(word,word_id)in zip(range(20),word2id_dict.items()):
    print(f"word {word}, its id {word_id}, its word freq {word2id_freq[word_id]}")


# **语料库的id表示**

# In[11]:


def convert_corpus_to_id(corpus,word2id_dict):
    corpus=[word2id_dict[word] for word in corpus]
    return corpus
corpus=convert_corpus_to_id(corpus,word2id_dict)
print(f"{len(corpus)} tokens in the corpus")
print(corpus[:50])


# **二次下采样**  
# 公式：  
# $$\ random_{num} < 1-\sqrt{\frac{10^{-4}} { wordid_{freq}} * wordall_{freq} }$$
# 其中$random_{num}$表示一个0-1之间的随机数，$wordid_{freq}$表示单词对应的频次，$wordall_{freq}$表示所有单词的频次之和   
# 
# 如果上式成立则丢弃，如果不成立则保留

# In[12]:


def subsampling(corpus,word2id_freq):
    # 函数discard判断一个词是否会被遗弃，如果TRUE则替换
    # 频次越大，被遗弃概率越大
    def discard(word_id):
        return random.uniform(0,1)<1-math.sqrt(
            1e-4/word2id_freq[word_id]*len(corpus)
        )
    corpus=[word for word in corpus if not discard(word)]
    return corpus
corpus=subsampling(corpus,word2id_freq)
print(f"{len(corpus)} tokens in the corpus")
print(corpus[:50])


# **构造数据集**

# In[13]:


# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料库
# negative_sample_num代表了对于每个正样本，我们需要随机采样的负样本用于训练
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但训练效果越慢
def build_data(corpus,max_window_size=3,negative_sample_num=4):
    # 使用一个list来存储处理好的数据
    dataset=[]
    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限，随机采样一个window_size
        window_size=random.randint(1,max_window_size)
        # 当前的window_size就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看做是正样本
        positive_word_range=(max(0,center_word_idx-window_size),
        min(len(corpus)-1,center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0],
        positive_word_range[1]+1) if idx !=center_word_idx]
    
        # 对每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（正样本，中心词，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((center_word,positive_word,1))
            
            # 开始负采样
            i=0
            while i<negative_sample_num:
                negative_word_candidate=random.randint(0,vocab_size-1)

                if negative_word_candidate not in positive_word_candidates:
                    # 把（负样本，中心词，label=0）的三元组放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((center_word,negative_word_candidate,0))
                    i+=1
    return dataset
# dataset=build_data(corpus,max_window_size=3,negative_sample_num=4)
# print(dataset[0])


# In[14]:


# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料库
# negative_sample_num代表了对于每个正样本，我们需要随机采样的负样本用于训练
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但训练效果越慢
def build_data_1(corpus,max_window_size=3,negative_sample_num=4):
    # 使用一个list来存储处理好的数据
    dataset=[]
    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(len(corpus)):
        # 以max_window_size为上限，随机采样一个window_size
        window_size=random.randint(1,max_window_size)
        # 当前的window_size就是center_word_idx所指向的词
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看做是正样本
        positive_word_range=(max(0,center_word_idx-window_size),
        min(len(corpus)-1,center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0],
        positive_word_range[1]+1) if idx !=center_word_idx]
    
        # 对每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for positive_word in positive_word_candidates:
            # 首先把（正样本，中心词，label=1）的三元组数据放入dataset中，
            # 这里label=1表示这个样本是个正样本
            dataset.append((positive_word,center_word,1))
            
            # 开始负采样
            i=0
            while i<negative_sample_num:
                negative_word_candidata=random.randint(0,vocab_size-1)

                if negative_word_candidata not in positive_word_candidates:
                    # 把（负样本，中心词，label=0）的三元组放入dataset中，
                    # 这里label=0表示这个样本是个负样本
                    dataset.append((negative_word_candidata,center_word,0))
                    i+=1
        # print(len(dataset))
    return dataset
# dataset=build_data_1(corpus,max_window_size=3,negative_sample_num=4)
# print(dataset[0])
# print(len(dataset))


# In[15]:


# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料库
# negative_sample_num代表了对于每个正样本，我们需要随机采样的负样本用于训练
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但训练效果越慢
def build_data_2(corpus,window_size=3):
    # 使用一个list来存储处理好的数据
    dataset=[]
    # 从左到右，开始枚举每个中心点的位置
    for center_word_idx in range(window_size,len(corpus)-window_size):
        center_word = corpus[center_word_idx]

        # 以当前中心词为中心，左右两侧在window_size内的词都可以看做是正样本
        positive_word_range=(center_word_idx-window_size,center_word_idx + window_size)
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0],
        positive_word_range[1]+1) if idx !=center_word_idx]
        dataset.append([positive_word_candidates,[center_word]])
    
        
    return dataset
# dataset=build_data_2(corpus,window_size=2)
# # print(dataset.shape)
# print(len(corpus))
# print(dataset[10])


# In[16]:


def build_batch(dataset,batch_size,epoch_num):
    
    center_word_batch=[]
    target_word_batch=[]
    label_batch=[]

    for epoch in range(epoch_num):
        random.shuffle(dataset)
        
        for center_word,target_word,label in dataset:
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            if len(center_word_batch)==batch_size:
                yield np.array(center_word_batch).astype('int64'),                    np.array(target_word_batch).astype('int64'),                    np.array(label_batch).astype('float32')
                center_word_batch=[]
                target_word_batch=[]
                label_batch=[]
    if len(center_word_batch)>0:
        yield np.array(center_word_batch).astype('int64'),            np.array(target_word_batch).astype('int64'),            np.array(label_batch).astype('float32')


# In[17]:


def build_batch_1(dataset,batch_size):
    
    def reader():
        random.shuffle(dataset)
        center_word_batch=[]
        target_word_batch=[]
        label_batch=[]
        for target_word,center_word,label in dataset:
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            if len(center_word_batch)==batch_size:
                yield np.array(center_word_batch).astype('int64'),                    np.array(target_word_batch).astype('int64'),                    np.array(label_batch).astype('float32')
                center_word_batch=[]
                target_word_batch=[]
                label_batch=[]
        if len(center_word_batch)>0:
            yield np.array(center_word_batch).astype('int64'),                np.array(target_word_batch).astype('int64'),                np.array(label_batch).astype('float32')
    return reader


# In[18]:


def build_batch_2(dataset,batch_size):
    def reader():
        random.shuffle(dataset)
        positive_word_batch=[]
        center_word_batch=[]
        for positive_word,center_word in dataset:
            positive_word_batch.append(positive_word)
            center_word_batch.append(center_word)

            if len(positive_word_batch)==batch_size:
                yield np.array(positive_word_batch).astype('int64'),                    np.array(center_word_batch).astype('int64')
                positive_word_batch=[]
                center_word_batch=[]
        if len(positive_word_batch)>0:
            yield np.array(positive_word_batch).astype('int64'),                np.array(center_word_batch).astype('int64')
    return reader
# data_reader=build_batch_2(dataset,batch_size=16)


# In[19]:


class CBOW(nn.Layer):
    def __init__(self,vocab_size,embed_size,window_size=5):
        super(CBOW,self).__init__()
        self.positive_num=2*window_size
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.l1=nn.Linear(embed_size,128)
        self.l2=nn.Linear(128,vocab_size)
    def forward(self,x_sample):
        sum_hidden=0
        for x in x_sample:
            inputs=self.embed(x)
            inputs=nn.Flatten()(inputs)
            inputs=nn.ReLU()(inputs)
            sum_hidden+=self.l1(inputs)
        avg_hidden=sum_hidden/self.positive_num
        out=self.l2(avg_hidden)
        out=F.log_softmax(out,axis=-1)
        return out

# model=CBOW(vocab_size,32,4)
# paddle.summary(model,(10,))        


# In[20]:


class CBOW_1(nn.Layer):
    def __init__(self,vocab_size,embed_size):
        super(CBOW_1,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.embedding_out=nn.Embedding(vocab_size,embed_size)
    def forward(self,target_words,center_words,labels):
        
        target_words_emb=self.embedding(target_words)
        center_words_emb=self.embedding_out(center_words)
        
        word_sim=paddle.multiply(target_words_emb,center_words_emb)
        word_sim=paddle.sum(word_sim,axis=-1)
        word_sim=paddle.reshape(word_sim,shape=[-1])
        pred=F.sigmoid(word_sim)

        loss=F.binary_cross_entropy_with_logits(word_sim,labels)
        loss=paddle.mean(loss)

        return pred,loss

# model=CBOW(vocab_size,32,4)
# paddle.summary(model,(10,))        


# In[21]:


class CBOW_2(nn.Layer):
    def __init__(self,vocab_size,embed_size,window_size=5):
        super(CBOW_2,self).__init__()
        self.positive_num=2*window_size
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.l1=nn.Linear(2*window_size*embed_size,128)
        self.l2=nn.Linear(128,vocab_size)

    def forward(self,x_sample):
        inputs=self.embed(x_sample)
        inputs=paddle.flatten(inputs)
        out=self.l1(inputs)
        out=F.relu(out)
        out=self.l2(out)
        out=F.log_softmax(out,axis=-1)
        return out

model=CBOW_2(vocab_size,32,2)
print(model(paddle.to_tensor([2,3,4,5])))  


# In[22]:


def get_similar_tokens(query_token,k,embed):
    W=embed.numpy()
    x=W[word2id_dict[query_token]]
    cos=np.dot(W,x)/np.sqrt(np.sum(W*W,axis=1)*np.sum(x*x)+1e-9)
    flat=cos.flatten()
    indices=np.argpartition(flat,-k)[-k:]
    indices=indices[np.argsort(-flat[indices])]
    similar_words=''
    for i in indices:
        if similar_words=='':
            similar_words=str(id2word_dict[i])
        else:
            similar_words+=','+str(id2word_dict[i])
    print("for word %s, the similar word is %s"%(query_token,similar_words))


# In[26]:


def train_2():
    # 超参数
    batch_size=128
    epoch_num=10
    embedding_size=200
    learning_rate=0.01
    window_size=3
    split_rate=0.01
    model_path='./work/CBOW_word2vec.pdmodel'
    load_model=True
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
    print("*"*15)
    print("data import ")
    print("*"*15)
    # 加载数据集
    corpus=load_text8()
    # 转换小写，切分单词
    corpus=data_preprocess(corpus)
    print('corpus:',len(corpus))
    # 切分数据集
    corpus=corpus[:int(len(corpus)*split_rate)]
    print('corpus:',len(corpus))
    # 构造词典
    word2id_dict,word2id_freq,id2word_dict=build_dict(corpus)
    vocab_size=len(word2id_dict)
    print('vocab_size:',vocab_size)
    # 语料库id表示
    corpus=convert_corpus_to_id(corpus,word2id_dict)
    print('corpus:',len(corpus))
    # 语料库下采样
    corpus=subsampling(corpus,word2id_freq)
    print('corpus:',len(corpus))
    # 构造数据集
    dataset=build_data_2(corpus,window_size=window_size)
    print('dataset:',len(dataset),dataset[10])
    # 设置或者加载模型
    if os.path.exists(model_path) and load_model:
        CBOW_model=paddle.load(model_path)
    else:
        CBOW_model=CBOW_2(vocab_size,embed_size=embedding_size,window_size=window_size)
    # 设置优化器
    opt=paddle.optimizer.Adam(learning_rate=learning_rate,parameters=CBOW_model.parameters())
    # 选择模型模式
    CBOW_model.train()
    print("start training ")
    print('*'*15)
    for epoch in range(epoch_num):
            # 加载数据生成器
        data_reader=build_batch_2(dataset,batch_size=batch_size)
        print(f'epoch:{epoch}')
        print('*'*15)
        for batch_id,(positive_samples,center_samples) in enumerate(data_reader()):
            positive_samples_vec=paddle.to_tensor(positive_samples)
            center_samples_vec=paddle.to_tensor(center_samples)
            loss_list=[]
            for positive_sample_vec,center_sample_vec in zip(positive_samples_vec,center_samples_vec):
                out=CBOW_model(positive_sample_vec)
                # print('out:',out.shape,out)
                loss=nn.functional.cross_entropy(out,center_sample_vec)
                # print('loss:',loss)
                loss=paddle.mean(loss)
                opt.clear_grad()
                loss.backward()
                opt.step()
                loss_list.append(loss.numpy())
            if batch_id %100==0:
                print("epoch:{}:batch_id:{}===loss:{:.6f}"                .format(epoch,batch_id,np.mean(loss_list)))
        print('*'*15)
    paddle.save(CBOW_model,model_path)
            

train_2()


# In[27]:


model_path='./work/CBOW_word2vec.pdmodel'
CBOW_model=paddle.load(model_path)
get_similar_tokens('one',5,model.embed.weight)
get_similar_tokens('year',5,model.embed.weight)
get_similar_tokens('what',5,model.embed.weight)
get_similar_tokens('in',5,model.embed.weight)
get_similar_tokens('if',5,model.embed.weight)
get_similar_tokens('soft',5,model.embed.weight)






# In[ ]:




