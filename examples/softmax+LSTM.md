## 1、基于层次softmax的优化策略

2013年，Mikolov提出的经典word2vec算法就是通过上下文来学习语义信息。word2vec包含两个经典模型：CBOW（Continuous Bag-of-Words）和Skip-gram。CBOW通过上下文的词向量推理中心词。而Skip-gram则根据中心词推理上下文。

输出时需要结果softmax函数归一化，得到对中心词的推理概率
$$
𝑠𝑜𝑓𝑡𝑚𝑎𝑥({O_i})= \frac{exp({O_i})}{\sum_jexp({O_j})}
$$
从上面的公式可以看出，softmax分母那项归一化，每次需要计算所有的O的输出值，才可以得到当前j节点的输出，当V很大的时候，O(V)的计算代价会非常高。所以在训练word2vec模型的时候，用到了两个tricks，一个是negative sampling，每次只采少量的负样本，不需要计算全部的V；

另外一个是hierarchical softmax，通过构建赫夫曼tree来做层级softmax，从复杂度O(V)降低到O(log_2V)

### **hierachical softmax**

#### 1、哈夫曼树

哈夫曼树( Huffman tree) 又称最优二叉树，是种带权路径最短的树，其特点是**权重越大，叶子节点就越靠近根节点，即权重越大，叶子节点搜索路径越短**

根据这个特性构造出的层次Softmax能够缩短目标类别的搜索路径。

![img](https://img-blog.csdnimg.cn/20200517121425833.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0JHb29kSGFiaXQ=,size_16,color_FFFFFF,t_70)



首先对所有在V词表的词，根据词频来构建哈夫曼tree，词频越大，路径越短，编码信息更少。tree中的所有的叶子节点构成了词V，中间节点则共有V-1个，上面的每个叶子节点存在唯一的从根到该节点的path，如上图所示，词**w_2**的path n(**w_2**,1 ) ,n(**w_2**,2) , n(**w_3**,3) 其中n(w,j)表示词w的path的第j个节点。

#### 2、叶子节点词的概率表示

上图假设我们需要计算**w_2**的输出概率，我们定义从根节点开始，每次经过中间节点，做一个二分类任务（左边或者右边），所以我们定义中间节点的n左边概率为：
$$
p(n,left) = σ({v_n^{'}}^T.h)
$$
其中v_n'是中间节点的向量，右边概率：
$$
p(n, right)=1-σ({v_n^{'}}^T.h)= σ(-{v_n^{'}}^T.h)
$$
从根节点到**w_2**，我们可以计算概率值为：
$$
p(w_2=w_O)=
p(n(w_2,1),left).p(n(w_2,2),left).p(n(w_3,3),right)=
σ({v_{n(w_2,1)}^{'}}^T.h). σ({v_{n(w_2,2)}^{'}}^T.h). σ(-{v_{n(w_3,3)}^{'}}^T.h)
$$
其中 σ为sigmoid函数

#### 3、 各叶子节点概率值相加为1

可以得出
$$
\sum_{i=1}^{V}p(w_i=w_O) = 1
$$

### 训练

 **1、预处理：构建haffman树**
根据语料中的每个word的词频构建赫夫曼tree，词频越高，则离树根越近，路径越短。如上图所示，词典V VV中的每个word都在叶子节点上，每个word需要计算两个信息：路径（经过的每个中间节点）以及赫夫曼编码，构建完赫夫曼tree后，每个叶子节点都有唯一的路径和编码，hierarchical softmax与softmax不同的是，在hierarchical softmax中，不对V中的word词进行向量学习，而是对中间节点进行向量学习，而每个叶子上的节点可以通过路径中经过的中间节点去表示。

**2 模型的输入**
输入部分，在cbow或者skip-gram模型，要么是上下文word对应的id词向量平均，要么是中心词对应的id向量，作为hidden层的输出向量

**3 样本label**
不同softmax的是，每个词word对应的是一个V大小的one-hot label，hierarchical softmax中每个叶子节点word，对应的label是赫夫曼编码，一般长度不超过 log_2V，在训练的时候，每个叶子节点的label统一编码到一个固定的长度，不足的可以进行pad

## 2、LSTM

### 前向传播算法

一个时刻的前向传播过程如下，对比RNN多了12个参数

![img](https://img-blog.csdnimg.cn/20190429152237896.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2doajc4NjExMA==,size_16,color_FFFFFF,t_70)

- 输入门：控制有多少输入信号会被融合。

$$
i_{t}=sigmoid(W_{i}X_{t}+V_{i}H_{t-1}+b_i) 
$$

- 遗忘门：控制有多少过去的记忆会被遗忘。
  $$
  f_{t}=sigmoid(W_{f}X_{t}+V_{f}H_{t-1}+b_f)
  $$
  
- 输出门：控制最终输出多少融合了记忆的信息

$$
o_{t}=sigmoid(W_{o}X_{t}+V_{o}H_{t-1}+b_o)
$$

- 单元状态：输入信号和过去的输入信号做一个信息融合。

$$
g_{t}=tanh(W_{g}X_{t}+V_{g}H_{t-1}+b_g)
$$

通过学习这些门的权重设置，长短时记忆网络可以根据当前的输入信号和记忆信息，有选择性地忽略或者强化当前的记忆或是输入信号，帮助网络更好地学习长句子的语义信息：

- 记忆信号：
  $$
  c_{t} = f_{t} \cdot c_{t-1} + i_{t} \cdot g_{t}
  $$
  
- 输出信号：
  $$
  h_{t} = o_{t} \cdot tanh(c_{t})
  $$

### 反向传播

反向传播通过梯度下降法迭代更新所有的参数            

长短期记忆神经网络的训练算法同样采用反向传播算法，主要有以下三个步骤：                                                                       

1.  前向计算每个神经元的输出值,对于长短期记忆神经网络来说，即 f_t 、 i_t 、c_t, O_t, h_t

    五个向量的值；

2.  反向计算每个神经元的误差项 delta 值。与循环神经网络一样，长短期记忆神经网络误差项的反向传播也是包括两个方向: 一个是沿时间轴的反向传播，即从当前 boldsymbol_t时刻开始，计算每个时刻的误差项; 一个是延网络层的反向传播，误差项向上一层传播，计算每一层的误差项；

3. 根据每个时刻的误差项，计算每个权重参数的误差梯度, 更新权重参数。



### litm词性分析

```
import torch
from torch import nn
from torch.autograd import Variable

# 给出两句话作为训练集，每个单词给出词性
train_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),  # DET:限定词，NN:名词，V：动词
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
print('======train_data=====')
print(train_data)

# 对单词给出数字编码，以便传入Embedding中转化成词向量
word_to_idx = {} # 对单词编码
tag_to_idx = {}# 对词性编码

for context, tag in train_data:  # context是句子，tag是后面的词性
    for word in context:  # 遍历每个单词，给每个单词编号
        if word.lower() not in word_to_idx:        # 如果该单词没有出现过
            word_to_idx[word.lower()] = len(word_to_idx)
            # lower()函数，将字符串中的所有大写字母转换为小写字母
            # 对该单词进行编号，从0开始
    for label in tag:  # 给每个词性标label，以及编号
        if label.lower() not in tag_to_idx:  # 如果该词性没有出现过
            tag_to_idx[label.lower()] = len(tag_to_idx)  # 对该词性进行编号，从0开始

#  定义编号和tag的字典，方便测试时使用，能够通过查找编号找到tag
idx_to_tag = {tag_to_idx[tag.lower()]: tag for tag in tag_to_idx}

# 对a-z的字符进行数字编码
alphabet = 'abcdefghijklmnopqrstuvwxyz'
character_to_idx = {}
for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i

# 这三个编码之后，用字典这种容器存储，每个元素对应一个数字编号
print('=====字符对应数字=====')
print(tag_to_idx)  # len=3
print(idx_to_tag)  # len=3
print(word_to_idx)  # len=8
print(character_to_idx)  # len=26


# 字符编码，将传入字符x中对应的编码，转化成LongTensor类型
def make_sequence(x, dic):
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx


print('=====''make_sequence()函数输出结果查看''=====')
print(make_sequence('abcdef', character_to_idx).type())  #得到该字符串的数据类型
print(make_sequence('abcdef', character_to_idx).size())   #得到该字符串的大小
print(make_sequence('abcdef', character_to_idx))        #得到该字符串每个字符对应的编号
print(make_sequence(train_data[0][0], word_to_idx))     #得到The dog ate the apple这句话每个单词对应的编号


# 定义字母字符的LSTM
class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        # n_char：26个字母，char_dim：单词字母向量维度，char_hidden：字母LSTM的输出维度，
        super(char_lstm, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)  #26个字母编号映射到低维空间，加速运算
        self.char_lstm = nn.LSTM(char_dim, char_hidden)      #输入char_dim维，输出char_hidden维

    def forward(self, x):
        x = self.char_embedding(x)
        out, _ = self.char_lstm(x)  #得到输出和隐藏状态
        return out[-1]  # (batch, hidden_size)  out[-1]可以表示我们需要的状态


# 定义词性分析的LSTM
class lstm_tagger(nn.Module):
    # n_word：单词的数目，n_dim：单词向量维度，n_char和char_dim同理，char_hidden：字母LSTM的输出维度，
    # n_hidden：单词词性预测LSTM的输出维度，n_tag：输出的词性分类
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag):
                    # 8,26,10, 100, 50, 128,3
        super(lstm_tagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden)  # 词性分析LSTM输入：词向量维度数+字符LSTM输出维度数
        self.classify =nn.Linear(n_hidden, n_tag)

    # 字符增强，传入句子的同时作为序列的同时，还要传入句子中的单词，用word表示
    def forward(self, x, word):
        char = []
        for w in word:  # 对于每个单词，遍历字母，做字母字符的lstm
            char_list = make_sequence(w, character_to_idx)
            char_list = char_list.unsqueeze(1)  # (seq, batch, feature) 满足 lstm 输入条件
            # unsqueeze(1)在第二维度上增加一个维度
            char_infor = self.char_lstm(Variable(char_list))  # (batch, char_hidden)
            char.append(char_infor)  #每个单词的特征空间
        char = torch.stack(char, dim=0)  # (seq, batch, feature)
        x = self.word_embedding(x)  # (batch, seq, word_dim)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # 改变顺序，变成(seq, batch, word_dim)
        x = torch.cat((x, char), dim=2)  # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        x, _ = self.lstm(x)
        seq, batch, h = x.shape
        x = x.view(-1, h)  # 重新 reshape 进行分类线性层
        out = self.classify(x) #size(len, n_tag)
        return out


net = lstm_tagger(len(word_to_idx), len(character_to_idx), 10, 100, 50, 128, len(tag_to_idx))
# (n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

print('=====开始训练=====')
# 开始训练
for e in range(500):# 训练500轮
    train_loss = 0
    for word, tag in train_data: # 遍历数据集中的字符串和词性串
        # word ['The', 'dog', 'ate', 'the', 'apple']
        # tag ['DET', 'NN', 'V', 'DET', 'NN']
        word_list = make_sequence(word, word_to_idx).unsqueeze(0)  # 在第一维度上，添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(train_data)))

# 看看预测的结果
print('=====测试阶段=====')
net = net.eval() # 设置成测试模式

test_sent = 'Everybody ate the apple read the book' #测试的句子
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0) # 得到单词编号

test_set = test_sent.split()
out = net(Variable(test), test_set)
print('=====输出out======')
print(out)
print('out的size是： ', out.size())  # 输入的test_sent是7个单词，单词词有三种：det,nn,v，所以结果是torch.Size([7, 3])
print('每一行tensor的三个值代表着：', tag_to_idx)
# 最后可以得到一个7x3的tensor，因为最后一层的线性层没有使用 softmax，所以数值不太像一个概率，
# 但是每一行数值最大的就表示属于该类，可以看到第一个单词 'Everybody' 属于 nn，
# 第二个单词 'ate' 属于 v，第三个单词 'the' 属于det，第四个单词 'apple' 属于 nn，
# 所以得到的这个预测结果是正确的

print('=====测试结果=====')
for i in range(len(test_set)):
    pred_tag_idx = out[i].argmax().item()
    # out[i]表示out这个tensor的第i行，
    # argmax()找出这一行最大值所在的位置，
    # .item()方法将tensor类型的pred_tag_idx变为int类型，才可以用于字典查询索引
    pred_word = idx_to_tag[pred_tag_idx]
    print('这个单词是: ', test_set[i], '. 它的词性是: ', idx_to_tag[pred_tag_idx])
```

### 二、使用keras实现LSTM 情感分析

keras提供一个LSTM层，用它来构造和训练一个多对一的RNN。我们的网络吸收一个序列（词序列）并输出一个情感分析值（正或负）。
训练集源自于kaggle上情感分类竞赛，包含7000个短句 UMICH SI650
每个句子有一个值为1或0的分别用来代替正负情感的标签，这个标签就是我们将要学习预测的。

**导入所需库**

```
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
```

**探索性分析**
特别地想知道语料中有多少个独立的词以及每个句子包含多少个词：

```p
#Read training data and generate vocabulary

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
ftrain = open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), 'rb')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1
ftrain.close()
```

通过上述代码，我们可以得到语料的值
maxlen: 42
len(word_freqs): 2313
我们将单词总数量设为固定值，并把所有其他词看作字典外的词，这些词全部用伪词unk（unknown）替换，预测时候将未见的词进行替换
句子包含的单词数（maxlen）让我们可以设置一个固定的序列长度，并且用0进行补足短句，把更长的句子截短至合适的长度。
把VOCABULARY_SIZE设置为2002，即源于字典的2000个词，加上伪词UNK和填充词PAD（用来补足句子到固定长度的词）
这里把句子最大长度MAX_SENTENCE_LENGTH定为40

```
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40
```

下一步我们需要两个查询表，RNN的每一个输入行都是一个词序列索引，索引按训练集中词的使用频度从高到低排序。这两张查询表允许我们通过给定的词来查找索引以及通过给定的索引来查找词。

```
#1 is UNK, 0 is PAD
#We take MAX_FEATURES-1 featurs to accound for PAD
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in 
                enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}
```

**接着我们将序列转换成词索引序列**
**补足MAX_SENTENCE_LENGTH定义的词的长度**
**因为我们的输出标签是二分类（正负情感）**

```
#convert sentences to sequences
X = np.empty((num_recs, ), dtype=list)
y = np.zeros((num_recs, ))
i = 0
ftrain = open(os.path.join(DATA_DIR, "umich-sentiment-train.txt"), 'rb')
for line in ftrain:
    label, sentence = line.strip().split("\t")
    words = nltk.word_tokenize(sentence.decode("ascii", "ignore").lower())
    seqs = []
    for word in words:
        if word2index.has_key(word):
            seqs.append(word2index[word])
        else:
            seqs.append(word2index["UNK"])
    X[i] = seqs
    y[i] = int(label)
    i += 1
ftrain.close()

# Pad the sequences (left padded with zeros)
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
```

**划分测试集与训练集**

```
# Split input into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, 
                                                random_state=42)
print(Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)
```

**训练模型**

```
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
# 美伦批大小32
BATCH_SIZE = 32
# 网络训练10轮
NUM_EPOCHS = 10
# Build model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, 
                    input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(Dropout(0.2)))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", 
              metrics=["accuracy"])

history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, 
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, ytest))
```


**最后我们在测试集上评估模型并打印出评分和准确率**

```
# evaluate

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))

for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = " ".join([index2word[x] for x in xtest[0].tolist() if x != 0])
    print("%.0f\t%d\t%s" % (ypred, ylabel, sent))
```


至此我们使用keras实现lstm的情感分析实例
在此实例中可学习到keras框架的使用、lstm模型搭建、短语句处理方式
