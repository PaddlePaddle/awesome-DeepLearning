```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

CBOW的原理
2013年，Mikolov提出了经典的word2vec算法，该算法通过上下文来学习语义信息。word2vec包含两个经典模型，CBOW（Continuous Bag-of-Words）和Skip-gram.
我们重点介绍一下CBOW模型的原理：
举个例子：Two boys are playing basketball.
在这个句子中我们定'are'为中心词，则Two，boys，playing，basketball为上下文。CBOW模型的原理就是利用上下文来预测中心词，即利用Two，boys，playing，basketball预测中心词：are。这样一来，are的语义就会分别传入上下文的信息中。不难想到，经过大量样本训练，一些量词，比如one，two就会自动找到它们的同义词，因为它们都具有are等中心词的语义。
CBOW的算法实现
![](https://ai-studio-static-online.cdn.bcebos.com/e7f14a7c383942799d6e84cfc50d626ef9145126a1af44d5ae859feb06b32f47)
如 图1 所示，CBOW是一个具有3层结构的神经网络，分别是：
Input Layer（输入层）：接收one-hot张量 V∈R1×vocab_sizeV \in R^{1 \times \text{vocab\_size}}V∈R 1×vocab_size
  作为网络的输入，里面存储着当前句子中上下文单词的one-hot表示。
Hidden Layer（隐藏层）：将张量VVV乘以一个word embedding张量W1∈Rvocab_size×embed_sizeW^1 \in R^{\text{vocab\_size} \times \text{embed\_size}}W1∈Rvocab_size×embed_size，并把结果作为隐藏层的输出，得到一个形状为R1×embed_sizeR^{1 \times \text{embed\_size}}R 1×embed_size的张量，里面存储着当前句子上下文的词向量。
Output Layer（输出层）：将隐藏层的结果乘以另一个word embedding张量W2∈Rembed_size×vocab_sizeW^2 \in R^{\text{embed\_size} \times \text{vocab\_size}}W 2∈Rembed_size×vocab_size，得到一个形状为R1×vocab_sizeR^{1 \times \text{vocab\_size}}R 1×vocab_size的张量。这个张量经过softmax变换后，就得到了使用当前上下文对中心的预测结果。根据这个softmax的结果，我们就可以去训练词向量模型。
在实际操作中，使用一个滑动窗口（一般情况下，长度是奇数），从左到右开始扫描当前句子。每个扫描出来的片段被当成一个小句子，每个小句子中间的词被认为是中心词，其余的词被认为是这个中心词的上下文。
CBOW算法和skip-gram算法最本质的区别就是：CBOW算法是以上下文预测中心词，而skip-gram算法是以中心城预测上下文。


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
from paddle.nn import Embedding
import paddle.nn.functional as F
```


```python
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
def readdata():
    corpus_url = "text8.txt"
    with open(corpus_url, "r") as f:  # 打开文件
        corpus = f.read().strip("\n")  # 读取文件
        print(corpus)
    f.close()
    return corpus
corpus = readdata()

corpus[:250]
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.
    
    Current values:
    NotebookApp.iopub_data_rate_limit=1000000.0 (bytes/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    





    ' anarchism originated as a term of abuse first used against early working class radicals including the diggers of the english revolution and the sans culottes of the french revolution whilst the term is still used in a pejorative way to describe any '




```python
def data_preprocess(corpus):
    # 由于英文单词出现在句首的时候经常要大写，所以我们把所有英文字符都转换为小写，
    # 以便对语料进行归一化处理（Apple vs apple等）
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus

corpus = data_preprocess(corpus)
corpus[:50]
```




    ['anarchism',
     'originated',
     'as',
     'a',
     'term',
     'of',
     'abuse',
     'first',
     'used',
     'against',
     'early',
     'working',
     'class',
     'radicals',
     'including',
     'the',
     'diggers',
     'of',
     'the',
     'english',
     'revolution',
     'and',
     'the',
     'sans',
     'culottes',
     'of',
     'the',
     'french',
     'revolution',
     'whilst',
     'the',
     'term',
     'is',
     'still',
     'used',
     'in',
     'a',
     'pejorative',
     'way',
     'to',
     'describe',
     'any',
     'act',
     'that',
     'used',
     'violent',
     'means',
     'to',
     'destroy',
     'the']




```python
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
```


```python
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(10), word2id_dict.items()):
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



```python
def convert_corpus_to_id(corpus, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

corpus = convert_corpus_to_id(corpus, word2id_dict)
```


```python
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
```

    8745735 tokens in the corpus
    [5233, 3080, 194, 3133, 45, 155, 741, 476, 10571, 27349, 15067, 58112, 854, 3580, 194, 10712, 214, 1324, 454, 2731]



```python
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
```


```python
for _, (center_word, target_word, label) in zip(range(25), dataset):
    print("center_word %s, target %s, label %d" % (id2word_dict[center_word],
                                                   id2word_dict[target_word], label))
```

    center_word originated, target anarchism, label 1
    center_word originated, target swp, label 0
    center_word originated, target osteopaths, label 0
    center_word originated, target fmarchives, label 0
    center_word originated, target entlassung, label 0
    center_word term, target anarchism, label 1
    center_word term, target razija, label 0
    center_word term, target trofie, label 0
    center_word term, target nubia, label 0
    center_word term, target retief, label 0
    center_word originated, target term, label 1
    center_word originated, target yrr, label 0
    center_word originated, target stampa, label 0
    center_word originated, target multilineal, label 0
    center_word originated, target wrongheaded, label 0
    center_word abuse, target term, label 1
    center_word abuse, target weinbaum, label 0
    center_word abuse, target sismi, label 0
    center_word abuse, target kkr, label 0
    center_word abuse, target putabant, label 0
    center_word term, target abuse, label 1
    center_word term, target splendidly, label 0
    center_word term, target putrefaction, label 0
    center_word term, target gentilesse, label 0
    center_word term, target exciter, label 0



```python
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
```


```python
for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)
```

           [ 56]]))


```python
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
```


```python
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

# 通过我们定义的CBOW类，来构造一个Skip-gram模型网络
cbow_model = CBOW(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters = cbow_model.parameters())

# 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
for epoch_num, context_words, target_words, label, eval_words in build_batch(
    dataset, batch_size, epoch_num):
    # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
    context_words_var = paddle.to_tensor(context_words)
    target_words_var = paddle.to_tensor(target_words)
    label_var = paddle.to_tensor(label)
    eval_words_var = paddle.to_tensor(eval_words)
    
    # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
    pred, loss  = cbow_model(
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
        get_similar_tokens('movie', 5, cbow_model.embedding.weight)
        get_similar_tokens('one', 5, cbow_model.embedding.weight)
        get_similar_tokens('who', 5, cbow_model.embedding.weight)
```

    step 1000, loss 0.693
    step 2000, loss 0.687
    step 3000, loss 0.626
    step 4000, loss 0.513
    step 5000, loss 0.373
    step 6000, loss 0.305
    step 7000, loss 0.288
    step 8000, loss 0.214
    step 9000, loss 0.201
    step 10000, loss 0.203
    for word movie, the similar word is movie
    for word movie, the similar word is processing
    for word movie, the similar word is someone
    for word movie, the similar word is knew
    for word movie, the similar word is boundary
    for word one, the similar word is one
    for word one, the similar word is selection
    for word one, the similar word is names
    for word one, the similar word is acts
    for word one, the similar word is leaving
    for word who, the similar word is who
    for word who, the similar word is minutes
    for word who, the similar word is addition
    for word who, the similar word is standards
    for word who, the similar word is portion
    step 11000, loss 0.227

