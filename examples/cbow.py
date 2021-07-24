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


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[ ]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[20]:


# 首先导入后续会用到的飞桨包
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


# In[21]:


# 读取语料用来训练word2vec
def readdata():
    corpus_url = "data/data98805/text8.txt"
    with open(corpus_url, "r") as f:  # 打开文件
        corpus = f.read().strip("\n")  # 读取文件
        print(corpus)
    f.close()
    return corpus
corpus = readdata()


# In[22]:


# 打印前500个字符查看语料的格式
corpus[:250]


# In[23]:


def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus

corpus = data_preprocess(corpus)


# In[24]:


corpus[:10]


# 在经过切词后，需要对语料进行统计，为每个词构造ID。一般来说，可以根据每个词在语料中出现的频次构造ID，频次越高，ID越小，便于对词典进行管理。代码如下：


# In[25]:


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


# In[26]:


# 总共有多少的词 按照频率打印前十个进行查看
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(10), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))


# 得到word2id词典后，还需要进一步处理原始语料，把每个词替换成对应的ID，便于神经网络进行处理，代码如下：


# In[27]:


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

corpus = convert_corpus_to_id(corpus, word2id_dict)


# In[28]:


print("%d tokens in the corpus" % len(corpus))
print(corpus[:20])


# 接下来，需要使用二次采样法处理原始文本。二次采样法的主要思想是降低高频词在语料中出现的频次。方法是随机将高频的词抛弃，频率越高，被抛弃的概率就越大；频率越低，被抛弃的概率就越小。标点符号或冠词这样的高频词就会被抛弃，从而优化整个词表的词向量训练效果，代码如下：


# In[29]:


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
print(corpus[:20])


# 在完成语料数据预处理之后，需要构造训练数据。根据上面的描述，我们需要使用一个滑动窗口对语料从左到右扫描，在每个窗口内，中心词需要预测它的上下文，并形成训练数据。
# 
# 在实际操作中，由于词表往往很大（50000，100000等），对大词表的一些矩阵运算（如softmax）需要消耗巨大的资源，因此可以通过负采样的方式模拟softmax的结果。
# 
# 1. 给定一个中心词和一个需要预测的上下文词，把这个上下文词作为正样本。
# 2. 通过词表随机采样的方式，选择若干个负样本。
# 3. 把一个大规模分类问题转化为一个2分类问题，通过这种方式优化计算速度。


# In[30]:


# 构造数据，准备模型训练
# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。 
def build_data(corpus, word2id_dict, word2id_freq, max_window_size = 3, negative_sample_num = 4):
    
    #使用一个list存储处理好的数据
    dataset = []
    center_word_idx=0

    #从左到右，开始枚举每个中心点的位置
    while center_word_idx < len(corpus):
        #以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        #当前的中心词就是center_word_idx所指向的词，可以当作正样本
        positive_word = corpus[center_word_idx]

        #以当前中心词为中心，左右两侧在window_size内的词就是上下文
        context_word_range = (max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        context_word_candidates = [corpus[idx] for idx in range(context_word_range[0], context_word_range[1]+1) if idx != center_word_idx]

        #对于每个正样本来说，随机采样negative_sample_num个负样本，用于训练
        for context_word in context_word_candidates:
            #首先把（上下文，正样本，label=1）的三元组数据放入dataset中，
            #这里label=1表示这个样本是个正样本
            dataset.append((context_word, positive_word, 1))

            #开始负采样
            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size-1)

                if negative_word_candidate is not positive_word:
                    #把（上下文，负样本，label=0）的三元组数据放入dataset中，
                    #这里label=0表示这个样本是个负样本
                    dataset.append((context_word, negative_word_candidate, 0))
                    i += 1
        
        center_word_idx = min(len(corpus) - 1, center_word_idx + window_size)
        if center_word_idx == (len(corpus) - 1):
            center_word_idx += 1
    
    return dataset

corpus_light = corpus[:int(len(corpus)*0.2)]
dataset = build_data(corpus_light, word2id_dict, word2id_freq)


# In[31]:


for _, (center_word, target_word, label) in zip(range(25), dataset):
    print("center_word %s, target %s, label %d" % (id2word_dict[center_word],
                                                   id2word_dict[target_word], label))


# 训练数据准备好后，把训练数据都组装成mini-batch，并准备输入到网络中进行训练，代码如下：


# In[32]:


# 构造mini-batch，准备对模型进行训练
# 我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    
    #context_word_batch缓存batch_size个中心词
    context_word_batch = []
    #target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    #label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []
    #eval_word_batch每次随机生成几个样例，用于在运行阶段对模型做评估，以便更好地可视化训练效果。
    eval_word_batch = []
    

    for epoch in range(epoch_num):
        #每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for context_word, target_word, label in dataset:
            #遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            context_word_batch.append([context_word])
            target_word_batch.append([target_word])
            label_batch.append(label)
            
            #构造训练中评估的样本，这里我们生成'one','king','who'三个词的同义词，
            #看模型认为的同义词有哪些
            if len(eval_word_batch) == 0:
                eval_word_batch.append([word2id_dict['one']])
            elif len(eval_word_batch) == 1:
                eval_word_batch.append([word2id_dict['king']])
            elif len(eval_word_batch) ==2:
                eval_word_batch.append([word2id_dict['who']])

            #当样本积攒到一个batch_size后，我们把数据都返回回来
            #在这里我们使用numpy的array函数把list封装成tensor
            #并使用python的迭代器机制，将数据yield出来
            #使用迭代器的好处是可以节省内存
            if len(context_word_batch) == batch_size:
                yield epoch,                    np.array(context_word_batch).astype("int64"),                    np.array(target_word_batch).astype("int64"),                    np.array(label_batch).astype("float32"),                    np.array(eval_word_batch).astype("int64")
                context_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []
        
    if len(context_word_batch) > 0:
        yield epoch,            np.array(context_word_batch).astype("int64"),            np.array(target_word_batch).astype("int64"),            np.array(label_batch).astype("float32"),            np.array(eval_word_batch).astype("int64")


# In[33]:


for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)


# ### 2.2 定义CBOW网络结构
# 
# 定义CBOW的网络结构，用于模型训练。在飞桨动态图中，对于任意网络，都需要定义一个继承自paddle.nn.layer的类来搭建网络结构、参数等数据的声明。同时需要在forward函数中定义网络的计算逻辑。值得注意的是，我们仅需要定义网络的前向计算逻辑，飞桨会自动完成神经网络的后向计算。
# 
# 在CBOW的网络结构中，使用的最关键的APi是[paddle.nn.Embedding](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Embedding_cn.html)函数，可以用其实现Embedding的网络层。通过查询飞桨的API文档，可以得到如下更详细的说明：
# 
# paddle.nn.Embedding(numembeddings, embeddingdim, paddingidx=None, sparse=False, weightattr=None, name=None)
# 
# 该接口用于构建 Embedding 的一个可调用对象，其根据input中的id信息从embedding矩阵中查询对应embedding信息，并会根据输入的size (numembeddings, embeddingdim)自动构造一个二维embedding矩阵。 输出Tensor的shape是在输入Tensor shape的最后一维后面添加了emb_size的维度。注：input中的id必须满足 0 =< id < size[0]，否则程序会抛异常退出。


# In[35]:


#定义CBOW训练网络结构
#使用paddlepaddle的2.0.0版本
#一般来说，在使用paddle训练的时候，我们需要通过一个类来定义网络结构，这个类继承了paddle.nn.layer
class SkipGram(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_size, init_scale=0.1):
        # vocab_size定义了这个skipgram这个模型的词表大小
        # embedding_size定义了词向量的维度是多少
        # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(SkipGram, self).__init__()
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

        #我们通过一个矩阵乘法，来对每个词计算他的同义词
        #on_fly在机器学习或深度学习中往往指在在线计算中做什么，
        #比如我们需要在训练中做评估，就可以说evaluation_on_fly
        # word_sim_on_fly = paddle.matmul(eval_words_emb, 
        #     self.embedding._w, transpose_y = True)

        # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
        return pred, loss # , word_sim_on_fly


# ### 2.3 网络训练
# 
# 完成网络定义后，就可以启动模型训练。我们定义每隔100步打印一次Loss，以确保当前的网络是正常收敛的。
# 
# 同时，我们每隔10000步观察一下skip-gram计算出来的同义词（使用 embedding的乘积），可视化网络训练效果，代码如下：


# In[ ]:


# 开始训练，定义一些训练过程中需要使用的超参数
batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001

#定义一个使用word-embedding查询同义词的函数
#这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
#我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
#具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
#两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
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

# 通过我们定义的SkipGram类，来构造一个Skip-gram模型网络
skip_gram_model = SkipGram(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters = skip_gram_model.parameters())

# 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
for epoch_num, context_words, target_words, label, eval_words in build_batch(
    dataset, batch_size, epoch_num):
    # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
    context_words_var = paddle.to_tensor(context_words)
    target_words_var = paddle.to_tensor(target_words)
    label_var = paddle.to_tensor(label)
    eval_words_var = paddle.to_tensor(eval_words)
    
    # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
    pred, loss  = skip_gram_model(
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
    if step % 10000 ==0:
        get_similar_tokens('movie', 5, skip_gram_model.embedding.weight)
        get_similar_tokens('one', 5, skip_gram_model.embedding.weight)
        get_similar_tokens('who', 5, skip_gram_model.embedding.weight)


# ![](https://ai-studio-static-online.cdn.bcebos.com/379357ce4e37406fbda59aefb1e2ad166c35c9534f5d4a37bf1666f5d37af381)
# 
# 
# 从打印结果可以看到，经过一定步骤的训练，Loss逐渐下降并趋于稳定。
# 
# 同时也可以发现CBOW模型可以学习到一些有趣的语言现象。

# # 总结 
# 
# **CBOW**提供了一种根据上下文推理中心词的思路。
# 
# 比如在多数情况下，“香蕉”和“橘子”更加相似，而“香蕉”和“句子”就没有那么相似；同时，“香蕉”和“食物”、“水果”的相似程度可能介于“橘子”和“句子”之间。那么如何让存储的词向量具备这样的语义信息呢？
# 
# 我们先学习自然语言处理领域的一个小技巧。在自然语言处理研究中，科研人员通常有一个共识：使用一个单词的上下文来了解这个单词的语义，比如：
# 
# “苹果手机质量不错，就是价格有点贵。”
# 
# “这个苹果很好吃，非常脆。”
# 
# “菠萝质量也还行，但是不如苹果支持的APP多。”
# 
# 在上面的句子中，我们通过上下文可以推断出第一个“苹果”指的是苹果手机，第二个“苹果”指的是水果苹果，而第三个“菠萝”指的应该也是一个手机。事实上，在自然语言处理领域，使用上下文描述一个词语或者元素的语义是一个常见且有效的做法。我们可以使用同样的方式训练词向量，让这些词向量具备表示语义信息的能力。
# 
# 

