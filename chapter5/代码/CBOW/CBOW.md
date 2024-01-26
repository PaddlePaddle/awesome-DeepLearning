# CBOW
CBOW模型，利用上下文或周围的单词来预测中心词。**输入**：某一个特征词的上下文相关对应的词向量（单词的**one-hot编码**）；**输出：**这特定的一个词的词向量(单词的one-hot编码）。


```python
import io
import os
import sys
import requests
from collections import OrderedDict 
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.nn import Embedding
```

### 数据处理

首先，找到一个合适的语料用于训练word2vec模型。我们选择text8数据集，这个数据集里包含了大量从维基百科收集到的英文语料，我们可以通过如下码下载，下载后的文件被保存在当前目录的text8.txt文件内。


```python
#下载语料用来训练word2vec
def download():
    #可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    #使用python的requests包下载数据集到本地
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    #把下载后的文件存储在当前目录的text8.txt文件内
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()
download()
```


```python
#读取text8数据
def load_text8():
    with open("./text8.txt", "r") as f:
        corpus = f.read().strip("\n")
    f.close()

    return corpus

corpus = load_text8()

#打印前500个字符，简要看一下这个语料的样子
print(corpus[:500])
```

     anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any act that used violent means to destroy the organization of society it has also been taken up as a positive label by self defined anarchists the word anarchism is derived from the greek without archons ruler chief king anarchism as a political philoso



```python
#对语料进行预处理（分词）
def data_preprocess(corpus):
    #由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    #以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")

    return corpus

corpus = data_preprocess(corpus)
print(corpus[:50])
```

    ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early', 'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the', 'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term', 'is', 'still', 'used', 'in', 'a', 'pejorative', 'way', 'to', 'describe', 'any', 'act', 'that', 'used', 'violent', 'means', 'to', 'destroy', 'the']


在经过切词后，需要对语料进行统计，为每个词构造ID。一般来说，可以根据每个词在语料中出现的频次构造ID，频次越高，ID越小，便于对词典进行管理。代码如下：


```python
#构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    #首先统计每个不同词的频率（出现的次数），使用一个词典记录
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    #将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    #一般来说，出现频率高的高频词往往是：I，the，you这种代词，而出现频率低的词，往往是一些名词，如：nlp
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)
    
    #构造3个不同的词典，分别存储，
    #每个词到id的映射关系：word2id_dict
    #每个id出现的频率：word2id_freq
    #每个id到词典映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    #按照频率，从高到低，开始遍历每个单词，并为这个单词构造一个独一无二的id
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
```

    there are totoally 253854 different words in the corpus
    word the, its id 0, its word freq 1061396
    word of, its id 1, its word freq 593677
    word and, its id 2, its word freq 416629
    word one, its id 3, its word freq 411764
    word in, its id 4, its word freq 372201
    word a, its id 5, its word freq 325873
    word to, its id 6, its word freq 316376
    word zero, its id 7, its word freq 264975
    word nine, its id 8, its word freq 250430
    word two, its id 9, its word freq 192644
    word is, its id 10, its word freq 183153
    word as, its id 11, its word freq 131815
    word eight, its id 12, its word freq 125285
    word for, its id 13, its word freq 118445
    word s, its id 14, its word freq 116710
    word five, its id 15, its word freq 115789
    word three, its id 16, its word freq 114775
    word was, its id 17, its word freq 112807
    word by, its id 18, its word freq 111831
    word that, its id 19, its word freq 109510
    word four, its id 20, its word freq 108182
    word six, its id 21, its word freq 102145
    word seven, its id 22, its word freq 99683
    word with, its id 23, its word freq 95603
    word on, its id 24, its word freq 91250
    word are, its id 25, its word freq 76527
    word it, its id 26, its word freq 73334
    word from, its id 27, its word freq 72871
    word or, its id 28, its word freq 68945
    word his, its id 29, its word freq 62603
    word an, its id 30, its word freq 61925
    word be, its id 31, its word freq 61281
    word this, its id 32, its word freq 58832
    word which, its id 33, its word freq 54788
    word at, its id 34, its word freq 54576
    word he, its id 35, its word freq 53573
    word also, its id 36, its word freq 44358
    word not, its id 37, its word freq 44033
    word have, its id 38, its word freq 39712
    word were, its id 39, its word freq 39086
    word has, its id 40, its word freq 37866
    word but, its id 41, its word freq 35358
    word other, its id 42, its word freq 32433
    word their, its id 43, its word freq 31523
    word its, its id 44, its word freq 29567
    word first, its id 45, its word freq 28810
    word they, its id 46, its word freq 28553
    word some, its id 47, its word freq 28161
    word had, its id 48, its word freq 28100
    word all, its id 49, its word freq 26229



```python
#把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    #使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

corpus = convert_corpus_to_id(corpus, word2id_dict)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])
```

    17005207 tokens in the corpus
    [5233, 3080, 11, 5, 194, 1, 3133, 45, 58, 155, 127, 741, 476, 10571, 133, 0, 27349, 1, 0, 102, 854, 2, 0, 15067, 58112, 1, 0, 150, 854, 3580, 0, 194, 10, 190, 58, 4, 5, 10712, 214, 6, 1324, 104, 454, 19, 58, 2731, 362, 6, 3672, 0]


接下来，需要使用二次采样法处理原始文本。二次采样法的主要思想是降低高频词在语料中出现的频次，从而优化整个词表的词向量训练效果，代码如下：


```python
#使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    
    #这个discard函数决定了一个词会不会被替换，这个函数是具有随机性的，每次调用结果不同
    #如果一个词的频率很大，那么它被遗弃的概率就很大
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(
            1e-4 / word2id_freq[word_id] * len(corpus))

    corpus = [word for word in corpus if not discard(word)]
    return corpus

corpus = subsampling(corpus, word2id_freq)
print("%d tokens in the corpus" % len(corpus))
print(corpus[:50])
```

    8742951 tokens in the corpus
    [5233, 3080, 194, 3133, 58, 741, 476, 10571, 133, 27349, 854, 15067, 58112, 150, 3580, 10712, 214, 6, 1324, 2731, 362, 3672, 708, 371, 1423, 2757, 18, 567, 686, 7088, 5233, 1052, 320, 248, 44611, 2877, 792, 186, 5233, 602, 10, 1134, 2621, 8983, 279, 4147, 141, 6437, 4186, 5233]


在完成语料数据预处理之后，需要构造训练数据。根据上面的描述，我们需要使用一个滑动窗口对语料从左到右扫描，在每个窗口内，通过上下文预测中心词，并形成训练数据。

在实际操作中，由于词表往往很大，对大词表的一些矩阵运算需要消耗巨大的资源，因此可以通过负采样的方式模拟softmax的结果。具体来说，给定上下文和需要预测的中心词，把中心词作为正样本；通过词表随机采样的方式，选择若干个负样本。这样就把一个大规模分类问题转化为一个2分类问题，通过这种方式优化计算速度，代码如下：


```python
#构造数据，准备模型训练
#max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
#negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
#一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。 
def build_data(corpus, word2id_dict, word2id_freq, max_window_size = 3, 
               negative_sample_num = 4):
    
    #使用一个list存储处理好的数据
    dataset = []
    center_word_idx=0

    #从左到右，开始枚举每个中心点的位置
    while center_word_idx < len(corpus):
        #以max_window_size为上限，随机采样一个window_size，这样会使得训练更加稳定
        window_size = random.randint(1, max_window_size)
        #当前的中心词就是center_word_idx所指向的词，可以当作正样本
        positive_word = corpus[center_word_idx]

        #以当前中心词为中心，在其左右得到上下文
        context_word_range = (max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        context_word_candidates = [corpus[idx] for idx in range(context_word_range[0], context_word_range[1]+1) if idx != center_word_idx]

        #随机采样negative_sample_num个负样本，用于训练
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
        if center_word_idx % 100000 == 0:
            print(center_word_idx)
    
    return dataset

dataset = build_data(corpus, word2id_dict, word2id_freq)
for _, (context_word, target_word, label) in zip(range(50), dataset):
    print("center_word %s, target %s, label %d" % (id2word_dict[context_word],
                                                   id2word_dict[target_word], label))
```

    200000
    300000
    400000
    600000
    800000
    900000
    1100000
    1200000
    1500000
    1700000
    1800000
    1900000
    2300000
    2500000
    2900000
    3100000
    3200000
    3300000
    3400000
    3800000
    4200000
    4300000
    4400000
    4500000
    4600000
    4700000
    4900000
    5000000
    5100000
    5300000
    5500000
    5900000
    6300000
    6500000
    6600000
    6800000
    6900000
    7200000
    7300000
    7400000
    7500000
    7600000
    7800000
    8100000
    8200000
    8300000
    8400000
    8500000
    8600000
    center_word originated, target anarchism, label 1
    center_word originated, target privativum, label 0
    center_word originated, target tonnesen, label 0
    center_word originated, target pedagogue, label 0
    center_word originated, target gicas, label 0
    center_word term, target anarchism, label 1
    center_word term, target esperanton, label 0
    center_word term, target madrey, label 0
    center_word term, target moogles, label 0
    center_word term, target hemispheres, label 0
    center_word originated, target term, label 1
    center_word originated, target paullinus, label 0
    center_word originated, target doppelsterne, label 0
    center_word originated, target laon, label 0
    center_word originated, target lisbonne, label 0
    center_word abuse, target term, label 1
    center_word abuse, target styrs, label 0
    center_word abuse, target benfleet, label 0
    center_word abuse, target rimonabant, label 0
    center_word abuse, target marut, label 0
    center_word originated, target abuse, label 1
    center_word originated, target santar, label 0
    center_word originated, target gotische, label 0
    center_word originated, target fruzz, label 0
    center_word originated, target domr, label 0
    center_word term, target abuse, label 1
    center_word term, target aphrod, label 0
    center_word term, target octidi, label 0
    center_word term, target irpp, label 0
    center_word term, target ctf, label 0
    center_word used, target abuse, label 1
    center_word used, target arborway, label 0
    center_word used, target pacelle, label 0
    center_word used, target simpol, label 0
    center_word used, target macandrew, label 0
    center_word working, target abuse, label 1
    center_word working, target apsua, label 0
    center_word working, target chali, label 0
    center_word working, target tefilati, label 0
    center_word working, target nazizmu, label 0
    center_word term, target working, label 1
    center_word term, target voiron, label 0
    center_word term, target stoichiometric, label 0
    center_word term, target estevez, label 0
    center_word term, target swooses, label 0
    center_word abuse, target working, label 1
    center_word abuse, target devoir, label 0
    center_word abuse, target conservatorships, label 0
    center_word abuse, target yelland, label 0
    center_word abuse, target mangareviewer, label 0


 训练数据准备好后，把训练数据都组装成mini-batch，并准备输入到网络中进行训练，代码如下：


```python
#构造mini-batch，准备对模型进行训练
#我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
#并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    
    #context_word_batch缓存batch_size个中心词
    context_word_batch = []
    #target_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    target_word_batch = []
    #label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []
    ##评估
    eval_word_batch = []
    

    for epoch in range(epoch_num):
        #每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for context_word, target_word, label in dataset:
            context_word_batch.append([context_word])
            target_word_batch.append([target_word])
            label_batch.append(label)
            
            #构造训练中评估的样本，这里我们生成'one','king','chip'三个词的同义词，
            #看模型认为的同义词有哪些
            if len(eval_word_batch) == 0:
                eval_word_batch.append([word2id_dict['one']])
            elif len(eval_word_batch) == 1:
                eval_word_batch.append([word2id_dict['king']])
            elif len(eval_word_batch) ==2:
                eval_word_batch.append([word2id_dict['who']])
           

            if len(context_word_batch) == batch_size:
                yield epoch,\
                    np.array(context_word_batch).astype("int64"),\
                    np.array(target_word_batch).astype("int64"),\
                    np.array(label_batch).astype("float32"),\
                    np.array(eval_word_batch).astype("int64")
                context_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []
        
    if len(context_word_batch) > 0:
        yield epoch,\
            np.array(context_word_batch).astype("int64"),\
            np.array(target_word_batch).astype("int64"),\
            np.array(label_batch).astype("float32"),\
            np.array(eval_word_batch).astype("int64")

for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)
```

           [ 56]]))

### 网络定义

定义CBOW的网络结构，用于模型训练。代码如下：


```python
#一般来说，在使用fluid训练的时候，我们需要通过一个类来定义网络结构，这个类继承了fluid.dygraph.Layer
class CBOW(fluid.dygraph.Layer):
    def __init__(self, name_scope, vocab_size, embedding_size, init_scale=0.1):
        #name_scope定义了这个类某个具体实例的名字，以便于区分不同的实例（模型）
        #vocab_size定义了这个skipgram这个模型的词表大小
        #embedding_size定义了词向量的维度是多少
        #init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
        super(CBOW, self).__init__(name_scope)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        #使用paddle.fluid.dygraph提供的Embedding函数，构造一个词向量参数
        #这个参数的大小为：[self.vocab_size, self.embedding_size]
        #数据类型为：float32
        #这个参数的名称为：embedding_para
        #这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
        self.embedding = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))

        
        self.embedding_out = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5/embedding_size, high=0.5/embedding_size)))



    #定义网络的前向计算逻辑
    #context_words是一个tensor（mini-batch），表示中心词
    #target_words是一个tensor（mini-batch），表示目标词
    #label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
    #eval_words是一个tensor（mini-batch），
    #用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
    def forward(self, context_words, target_words, label, eval_words):
        #首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
        #这里context_words和eval_words_emb查询的是一个相同的参数
        #而target_words_emb查询的是另一个参数
        context_words_emb = self.embedding(context_words)
        target_words_emb = self.embedding_out(target_words)
        eval_words_emb = self.embedding(eval_words)
        
        #context_words_emb = [batch_size, embedding_size]
        #target_words_emb = [batch_size, embedding_size]
        word_sim = fluid.layers.elementwise_mul(context_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim = -1)
        pred = fluid.layers.sigmoid(word_sim)

        #通过估计的输出概率定义损失函数
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)
        

        word_sim_on_fly = fluid.layers.matmul(eval_words_emb, 
            self.embedding._w, transpose_y = True)

        return pred, loss, word_sim_on_fly
```

### 网络训练

完成网络定义后，就可以启动模型训练。我们定义每隔100步打印一次loss，以确保当前的网络是正常收敛的。同时，我们每隔1000步观察一下计算出来的同义词，可视化网络训练效果，代码如下：


```python
#开始训练，定义一些训练过程中需要使用的超参数
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

with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    skip_gram_model = CBOW("CBOW", vocab_size, embedding_size)
    #构造训练这个网络的优化器
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)

    #使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for epoch_num, context_words, target_words, label, eval_words in build_batch(
        dataset, batch_size, epoch_num):

    
        context_words_var = fluid.dygraph.to_variable(context_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)
        eval_words_var = fluid.dygraph.to_variable(eval_words)

        pred, loss, word_sim_on_fly = skip_gram_model(
            context_words_var, target_words_var, label_var, eval_words_var)

        #通过backward函数，让程序自动完成反向计算
        loss.backward()
        #通过minimize函数，让程序根据loss，完成一步对参数的优化更新
        adam.minimize(loss)
        #使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
        skip_gram_model.clear_gradients()

        #每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        step += 1
        if step % 100 == 0:
            print("epoch num:%d, step %d, loss %.3f" % (epoch_num, step, loss.numpy()[0]))


        if step % 1000 == 0:
            #           (curr_eval_word, ", ".join(top_n_sim_words)))
            get_similar_tokens('one', 5, skip_gram_model.embedding._w)
            get_similar_tokens('who', 5, skip_gram_model.embedding._w)
            get_similar_tokens('king', 5, skip_gram_model.embedding._w)
            
```


```python

```
