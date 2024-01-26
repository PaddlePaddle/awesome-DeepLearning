一、SoftMax
1.1 分层Softmax
      Hierarchical softmax （H-Softmax）是由Morin和Bengio受到二叉树的启发而提出。H-Softmax本质上是用层级关系替代了扁平化的softmax层，即每个叶子节点表示一个词语。
于是，计算单个词语概率值的过程被拆解为一系列的概率计算，这可以避免对所有词语进行标准化计算。用H-Softmax替换softmax层之后，词语的预测速度可以提升至少50倍，
速度的提升对于低延时要求的实时系统至关重要。
      我们可以把原来的softmax看做深度为1的树，词表中的每一个词语表示一个叶子节点。计算一个词语的softmax概率需要对节点的概率值做标准化。如果把softmax改为二叉树结构，
每个word表示叶子节点，那么只需要沿着通向该词语的叶子节点的路径搜索，而不需要考虑其它的节点。
     平衡二叉树的深度是log2(|V|)，因此，最多只需要计算log2(|V|)个节点就能得到目标词语的概率值。注意，得到的概率值已经经过了标准化，因为二叉树所有叶子节点组成一个概率分布，
所有叶子节点的概率值总和等于1。
     具体说来，当遍历树的时候，我们需要能够计算左侧分枝或右侧分枝的概率值。为此，给每个节点分配一个向量表示。与常规的softmax做法不同
这里不是给每个输出词语w生成词向量，而是给每个节点n计算一个向量。总共有|V|-1个节点，每个节点都有自己独一无二的向量表示，
H-Softmax方法用到的参数与常规的softmax几乎一样。于是，在给定上下文c时，就能够计算节点n左右两个分枝的概率。
     现在需要计算h与树的每个节点的向量v’n的内积，而不是与每个输出词语的向量计算。而且，现在只需要计算一个概率值，这里就是偏向n节点右枝的概率值。
     假设已知出现了词语“the”、“dog”、“and”、“the”，则出现词语“cat”的概率值就是在节点1向左偏的概率值、在节点2向右偏的概率以及在节点5向右偏的概率值的乘积。
Hugo Lachorelle在他的视频教程中给了更详细的解释。Rong[7]的文章也详细地解释了这些概念，并推导了H-Softmax。显然，树形的结构非常重要。若我们让模型在各个节点的预测更方便，
比如路径相近的节点概率值也相近，那么凭直觉系统的性能肯定还会提升。沿着这个思路，Morin和Bengio使用WordNet的同义词集作为树簇。然而性能依旧不如常规的softmax方法。
Mnih和Hinton[8]将聚类算法融入到树形结构的学习过程，递归地将词集分为两个集合，效果终于和softmax方法持平，计算量有所减小。
因为平衡二叉树的深度不超过log2(|V|)。若词表的大小是|V|=10000，那么搜索路径的平均长度就是13.3。即词表中的每个词语都能表示一个平均长度为13.3比特的向量，信息量为13.3比特。
    关于信息量：
在信息论中，人们习惯于将词语w概率值的负对数定义为信息量I(w)：
                                                                                                 I(w)=−log2p(w)
而熵H则是词表中所有词语的信息量的期望值：
                                                                                                H=∑i∈Vp(wi)I(wi)
熵也代表着根据信息的概率分布对信息编码所需要的最短平均编码长度。 抛硬币事件需要用1比特来编码正反两个时间，对于永恒不变的事件则只需0比特。
若用平衡二叉树的节点来表示词表中的词语，还是假设词表的大小|V|=10000，词表中词语的概率值均相等，那么熵H与平均搜索路径的长度恰好相等。
1.2 分片Softmax
     D-Softmax基于的假设是并不是所有词语都需要相同数量的参数：多次出现的高频词语需要更多的参数去拟合，而较少见的词语就可以用较少的参数。传统的softmax层用到了dx|V|
的稠密矩阵来存放输出的词向量表示v′w∈ℝd，采用了稀疏矩阵。他们将词向量v′w按照词频分块，每块区域的向量维度各不相同。分块数量和对应的维度是超参数，可以根据需要调整。
     A区域的词向量维度是dA（这个分块是高频词语，向量的维度较高），B和C区域的词向量维度分别是dB和dC。其余空白区域的值为0。隐藏层h的输出被视为是各个分块的级联，
h层的输出是由三个长度分别为dA、dB、dC的向量级联而成。D-Softmax只需计算各个向量段与h对应位置的内积，而不需整个矩阵和向量参与计算。
     由于大多数的词语只需要相对较少的参数，计算softmax的复杂度得到降低，训练速度因此提升。相对于H-Softmax方法，D-Softmax的优化方法在测试阶段仍然有效。
1.3 CNN-Softmax
     传统softmax层的另一种改进是受到Kim的论文启发，Kim对输入词向量vw采用了字符级别的CNN模型。相反，Jozefowicz在2016年将同样的方法用于输出词向量v′w，
并将这种方法称为CNN-Softmax。如果我们在输入端和输出端加上CNN模型，输出端CNN生成的向量v′w与输入端CNN生成的向量必然不相同，因为输入和输出的词向量矩阵就不一样。
     尽管这个方法仍需要计算常规softmax的标准化，但模型的参数却大大的减少：只需要保留CNN模型的参数，不需要保存整个词向量矩阵dx|V|。在测试阶段，输出词向量v′w可以提前计算，
所以性能损失不大。
    但是，由于字符串是在连续空间内表示，而且模型倾向于用平滑函数将字符串映射到词向量，因此基于字符的模型往往无法区分拼写相似但是含义不同的词语。为了消除上述影响，
增加了一个矫正因数，显著地缩小了CNN-Softmax与传统方法的性能差距。

二、LSTM实现机器翻译
环境设置
In [1]
import paddle
import paddle.nn.functional as F
import re
import numpy as np

print(paddle.__version__)
2.0.0-rc0
下载数据集
我们将使用 http://www.manythings.org/anki/ 提供的中英文的英汉句对作为数据集，来完成本任务。该数据集含有23610个中英文双语的句对。

In [2]
!wget -c https://www.manythings.org/anki/cmn-eng.zip && unzip cmn-eng.zip
--2020-11-02 17:07:29--  https://www.manythings.org/anki/cmn-eng.zip
Resolving www.manythings.org (www.manythings.org)... 104.24.108.196, 104.24.109.196, 172.67.173.198, ...
Connecting to www.manythings.org (www.manythings.org)|104.24.108.196|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1030722 (1007K) [application/zip]
Saving to: ‘cmn-eng.zip’

cmn-eng.zip         100%[===================>]   1007K   167KB/s    in 6.0s    

2020-11-02 17:07:36 (167 KB/s) - ‘cmn-eng.zip’ saved [1030722/1030722]

Archive:  cmn-eng.zip
  inflating: cmn.txt                 
  inflating: _about.txt              
In [3]
!wc -l cmn.txt
23610 cmn.txt
构建双语句对的数据结构
接下来我们通过处理下载下来的双语句对的文本文件，将双语句对读入到python的数据结构中。这里做了如下的处理。

对于英文，会把全部英文都变成小写，并只保留英文的单词。
对于中文，为了简便起见，未做分词，按照字做了切分。
为了后续的程序运行的更快，我们通过限制句子长度，和只保留部分英文单词开头的句子的方式，得到了一个较小的数据集。这样得到了一个有5508个句对的数据集。
In [4]
MAX_LEN = 10
In [5]
lines = open('cmn.txt', encoding='utf-8').read().strip().split('\n')
words_re = re.compile(r'\w+')

pairs = []
for l in lines:
    en_sent, cn_sent, _ = l.split('\t')
    pairs.append((words_re.findall(en_sent.lower()), list(cn_sent)))

# create a smaller dataset to make the demo process faster
filtered_pairs = []

for x in pairs:
    if len(x[0]) < MAX_LEN and len(x[1]) < MAX_LEN and \
    x[0][0] in ('i', 'you', 'he', 'she', 'we', 'they'):
        filtered_pairs.append(x)
           
print(len(filtered_pairs))
for x in filtered_pairs[:10]: print(x) 
5508
(['i', 'won'], ['我', '赢', '了', '。'])
(['he', 'ran'], ['他', '跑', '了', '。'])
(['i', 'quit'], ['我', '退', '出', '。'])
(['i', 'm', 'ok'], ['我', '沒', '事', '。'])
(['i', 'm', 'up'], ['我', '已', '经', '起', '来', '了', '。'])
(['we', 'try'], ['我', '们', '来', '试', '试', '。'])
(['he', 'came'], ['他', '来', '了', '。'])
(['he', 'runs'], ['他', '跑', '。'])
(['i', 'agree'], ['我', '同', '意', '。'])
(['i', 'm', 'ill'], ['我', '生', '病', '了', '。'])
创建词表
接下来我们分别创建中英文的词表，这两份词表会用来将英文和中文的句子转换为词的ID构成的序列。词表中还加入了如下三个特殊的词：

<pad>: 用来对较短的句子进行填充。
<bos>: "begin of sentence"， 表示句子的开始的特殊词。
<eos>: "end of sentence"， 表示句子的结束的特殊词。
Note: 在实际的任务中，可能还需要通过<unk>（或者<oov>）特殊词来表示未在词表中出现的词。

In [6]
en_vocab = {}
cn_vocab = {}

# create special token for pad, begin of sentence, end of sentence
en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2

en_idx, cn_idx = 3, 3
for en, cn in filtered_pairs:
    for w in en: 
        if w not in en_vocab: 
            en_vocab[w] = en_idx
            en_idx += 1
    for w in cn:  
        if w not in cn_vocab: 
            cn_vocab[w] = cn_idx
            cn_idx += 1

print(len(list(en_vocab)))
print(len(list(cn_vocab)))
2539
2039
创建padding过的数据集
接下来根据词表，我们将会创建一份实际的用于训练的用numpy array组织起来的数据集。

所有的句子都通过<pad>补充成为了长度相同的句子。
对于英文句子（源语言），我们将其反转了过来，这会带来更好的翻译的效果。
所创建的padded_cn_label_sents是训练过程中的预测的目标，即，每个中文的当前词去预测下一个词是什么词。
In [7]
padded_en_sents = []
padded_cn_sents = []
padded_cn_label_sents = []
for en, cn in filtered_pairs:
    # reverse source sentence
    padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
    padded_en_sent.reverse()
    padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
    padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1) 

    padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
    padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
    padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])

train_en_sents = np.array(padded_en_sents)
train_cn_sents = np.array(padded_cn_sents)
train_cn_label_sents = np.array(padded_cn_label_sents)

print(train_en_sents.shape)
print(train_cn_sents.shape)
print(train_cn_label_sents.shape)
(5508, 11)
(5508, 12)
(5508, 12)
创建网络
我们将会创建一个Encoder-AttentionDecoder架构的模型结构用来完成机器翻译任务。 首先我们将设置一些必要的网络结构中用到的参数。

In [8]
embedding_size = 128
hidden_size = 256
num_encoder_lstm_layers = 1
en_vocab_size = len(list(en_vocab))
cn_vocab_size = len(list(cn_vocab))
epochs = 20
batch_size = 16
Encoder部分
在编码器的部分，我们通过查找完Embedding之后接一个LSTM的方式构建一个对源语言编码的网络。飞桨的RNN系列的API，除了LSTM之外，还提供了SimleRNN, GRU供使用，同时，还可以使用反向RNN，双向RNN，多层RNN等形式。也可以通过dropout参数设置是否对多层RNN的中间层进行dropout处理，来防止过拟合。

除了使用序列到序列的RNN操作之外，也可以通过SimpleRNN, GRUCell, LSTMCell等API更灵活的创建单步的RNN计算，甚至通过继承RNNCellBase来实现自己的RNN计算单元。

In [9]
# encoder: simply learn representation of source sentence
class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = paddle.nn.Embedding(en_vocab_size, embedding_size,)
        self.lstm = paddle.nn.LSTM(input_size=embedding_size, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_encoder_lstm_layers)

    def forward(self, x):
        x = self.emb(x)
        x, (_, _) = self.lstm(x)
        return x
AttentionDecoder部分
在解码器部分，我们通过一个带有注意力机制的LSTM来完成解码。

单步的LSTM：在解码器的实现的部分，我们同样使用LSTM，与Encoder部分不同的是，下面的代码，每次只让LSTM往前计算一次。整体的recurrent部分，是在训练循环内完成的。
注意力机制：这里使用了一个由两个Linear组成的网络来完成注意力机制的计算，它用来计算出目标语言在每次翻译一个词的时候，需要对源语言当中的每个词需要赋予多少的权重。
对于第一次接触这样的网络结构来说，下面的代码在理解起来可能稍微有些复杂，你可以通过插入打印每个tensor在不同步骤时的形状的方式来更好的理解。
In [10]
# only move one step of LSTM, 
# the recurrent loop is implemented inside training loop
class AttentionDecoder(paddle.nn.Layer):
    def __init__(self):
        super(AttentionDecoder, self).__init__()
        self.emb = paddle.nn.Embedding(cn_vocab_size, embedding_size)
        self.lstm = paddle.nn.LSTM(input_size=embedding_size + hidden_size, 
                                   hidden_size=hidden_size)

        # for computing attention weights
        self.attention_linear1 = paddle.nn.Linear(hidden_size * 2, hidden_size)
        self.attention_linear2 = paddle.nn.Linear(hidden_size, 1)
        
        # for computing output logits
        self.outlinear =paddle.nn.Linear(hidden_size, cn_vocab_size)

    def forward(self, x, previous_hidden, previous_cell, encoder_outputs):
        x = self.emb(x)
        
        attention_inputs = paddle.concat((encoder_outputs, 
                                      paddle.tile(previous_hidden, repeat_times=[1, MAX_LEN+1, 1])),
                                      axis=-1
                                     )

        attention_hidden = self.attention_linear1(attention_inputs)
        attention_hidden = F.tanh(attention_hidden)
        attention_logits = self.attention_linear2(attention_hidden)
        attention_logits = paddle.squeeze(attention_logits)

        attention_weights = F.softmax(attention_logits)        
        attention_weights = paddle.expand_as(paddle.unsqueeze(attention_weights, -1), 
                                             encoder_outputs)

        context_vector = paddle.multiply(encoder_outputs, attention_weights)               
        context_vector = paddle.sum(context_vector, 1)
        context_vector = paddle.unsqueeze(context_vector, 1)
        
        lstm_input = paddle.concat((x, context_vector), axis=-1)

        # LSTM requirement to previous hidden/state: 
        # (number_of_layers * direction, batch, hidden)
        previous_hidden = paddle.transpose(previous_hidden, [1, 0, 2])
        previous_cell = paddle.transpose(previous_cell, [1, 0, 2])
        
        x, (hidden, cell) = self.lstm(lstm_input, (previous_hidden, previous_cell))
        
        # change the return to (batch, number_of_layers * direction, hidden)
        hidden = paddle.transpose(hidden, [1, 0, 2])
        cell = paddle.transpose(cell, [1, 0, 2])

        output = self.outlinear(hidden)
        output = paddle.squeeze(output)
        return output, (hidden, cell)
训练模型
接下来我们开始训练模型。

在每个epoch开始之前，我们对训练数据进行了随机打乱。
我们通过多次调用atten_decoder，在这里实现了解码时的recurrent循环。
teacher forcing策略: 在每次解码下一个词时，我们给定了训练数据当中的真实词作为了预测下一个词时的输入。相应的，你也可以尝试用模型预测的结果作为下一个词的输入。（或者混合使用）
In [11]
encoder = Encoder()
atten_decoder = AttentionDecoder()

opt = paddle.optimizer.Adam(learning_rate=0.001, 
                            parameters=encoder.parameters()+atten_decoder.parameters())

for epoch in range(epochs):
    print("epoch:{}".format(epoch))

    # shuffle training data
    perm = np.random.permutation(len(train_en_sents))
    train_en_sents_shuffled = train_en_sents[perm]
    train_cn_sents_shuffled = train_cn_sents[perm]
    train_cn_label_sents_shuffled = train_cn_label_sents[perm]

    for iteration in range(train_en_sents_shuffled.shape[0] // batch_size):
        x_data = train_en_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]
        sent = paddle.to_tensor(x_data)
        en_repr = encoder(sent)

        x_cn_data = train_cn_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]
        x_cn_label_data = train_cn_label_sents_shuffled[(batch_size*iteration):(batch_size*(iteration+1))]

        # shape: (batch,  num_layer(=1 here) * num_of_direction(=1 here), hidden_size)
        hidden = paddle.zeros([batch_size, 1, hidden_size])
        cell = paddle.zeros([batch_size, 1, hidden_size])

        loss = paddle.zeros([1])
        # the decoder recurrent loop mentioned above
        for i in range(MAX_LEN + 2):
            cn_word = paddle.to_tensor(x_cn_data[:,i:i+1])
            cn_word_label = paddle.to_tensor(x_cn_label_data[:,i])

            logits, (hidden, cell) = atten_decoder(cn_word, hidden, cell, en_repr)
            step_loss = F.cross_entropy(logits, cn_word_label)
            loss += step_loss

        loss = loss / (MAX_LEN + 2)
        if(iteration % 200 == 0):
            print("iter {}, loss:{}".format(iteration, loss.numpy()))

        loss.backward()
        opt.step()
        opt.clear_grad()
epoch:0
iter 0, loss:[7.6281333]
iter 200, loss:[3.3122177]
epoch:1
iter 0, loss:[3.1625233]
iter 200, loss:[3.3523889]
epoch:2
iter 0, loss:[2.868197]
iter 200, loss:[2.4968011]
epoch:3
iter 0, loss:[2.586331]
iter 200, loss:[2.4894257]
epoch:4
iter 0, loss:[2.6632512]
iter 200, loss:[2.320348]
epoch:5
iter 0, loss:[2.4188473]
iter 200, loss:[2.6118374]
epoch:6
iter 0, loss:[1.8900026]
iter 200, loss:[2.1481352]
epoch:7
iter 0, loss:[1.9027576]
iter 200, loss:[1.8338045]
epoch:8
iter 0, loss:[1.7218149]
iter 200, loss:[1.6443458]
epoch:9
iter 0, loss:[1.8346084]
iter 200, loss:[1.7748606]
epoch:10
iter 0, loss:[1.3841861]
iter 200, loss:[1.4972442]
epoch:11
iter 0, loss:[0.99217004]
iter 200, loss:[1.1514312]
epoch:12
iter 0, loss:[1.1948762]
iter 200, loss:[1.2036713]
epoch:13
iter 0, loss:[1.1935998]
iter 200, loss:[1.0770215]
epoch:14
iter 0, loss:[0.8845991]
iter 200, loss:[1.096318]
epoch:15
iter 0, loss:[0.6377462]
iter 200, loss:[0.8782622]
epoch:16
iter 0, loss:[0.9234778]
iter 200, loss:[0.72928995]
epoch:17
iter 0, loss:[0.5583935]
iter 200, loss:[0.9392061]
epoch:18
iter 0, loss:[0.5447843]
iter 200, loss:[0.69262314]
epoch:19
iter 0, loss:[0.70543337]
iter 200, loss:[0.6030283]
使用模型进行机器翻译
根据你所使用的计算设备的不同，上面的训练过程可能需要不等的时间。（在一台Mac笔记本上，大约耗时15~20分钟） 完成上面的模型训练之后，我们可以得到一个能够从英文翻译成中文的机器翻译模型。接下来我们通过一个greedy search来实现使用该模型完成实际的机器翻译。（实际的任务中，你可能需要用beam search算法来提升效果）

In [12]
encoder.eval()
atten_decoder.eval()

num_of_exampels_to_evaluate = 10

indices = np.random.choice(len(train_en_sents),  num_of_exampels_to_evaluate, replace=False)
x_data = train_en_sents[indices]
sent = paddle.to_tensor(x_data)
en_repr = encoder(sent)

word = np.array(
    [[cn_vocab['<bos>']]] * num_of_exampels_to_evaluate
)
word = paddle.to_tensor(word)

hidden = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])
cell = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])

decoded_sent = []
for i in range(MAX_LEN + 2):
    logits, (hidden, cell) = atten_decoder(word, hidden, cell, en_repr)
    word = paddle.argmax(logits, axis=1)
    decoded_sent.append(word.numpy())
    word = paddle.unsqueeze(word, axis=-1)
    
results = np.stack(decoded_sent, axis=1)
for i in range(num_of_exampels_to_evaluate):
    en_input = " ".join(filtered_pairs[indices[i]][0])
    ground_truth_translate = "".join(filtered_pairs[indices[i]][1])
    model_translate = ""
    for k in results[i]:
        w = list(cn_vocab)[k]
        if w != '<pad>' and w != '<eos>':
            model_translate += w
    print(en_input)
    print("true: {}".format(ground_truth_translate))
    print("pred: {}".format(model_translate))
you have to leave
true: 你們得走了。
pred: 你們得走了。
i hope he will wait for me
true: 我希望他會等我。
pred: 我希望他會來我。
we took a walk along the river
true: 我們沿著河散步。
pred: 我們輪流開車去。
you seem to be waiting for somebody
true: 你看來在等人。
pred: 你看起來看她。
i gave my cold to him
true: 我的感冒传染了他.
pred: 我的信我的弄纪。
i don t like this one
true: 我不喜欢这个。
pred: 我不喜欢这个。
you broke the rules
true: 你触犯了规则。
pred: 你触犯了规则。
he is just my age
true: 他和我同岁。
pred: 他和我同岁。
they sell sporting goods
true: 他们卖体育产品。
pred: 他們被嚇到了。
i didn t tell them
true: 我沒告訴他們。
pred: 我不觉得他诚实。

二、LSTM实现动态手势识别
项目介绍：基于TSN_LSTM的动态手势识别

Twentybn手势识别数据集介绍：
数据集链接：https://20bn.com/customers/sign_in
数据集总共包含十万多个视频（图片形式），总共27种动态手势
TSN模型：
基于长范围时间结构（long-range temporal structure）建模，结合了稀疏时间采样策略（sparse temporal sampling strategy）和视频级监督（video-level supervision）来保证使用整段视频时学习得有效和高效。two-stream 卷积网络对于长范围时间结构的建模无能为力，
主要因为它仅仅操作一帧（空间网络）或者操作短片段中的单堆帧（时间网络），因此对时间上下文的访问是有限的。视频级框架TSN可以从整段视频中建模动作。和two-stream一样，TSN也是由空间流卷积网络和时间流卷积网络构成。但不同于two-stream采用单帧或者单堆帧，
TSN使用从整个视频中稀疏地采样一系列短片段，每个片段都将给出其本身对于行为类别的初步预测，从这些片段的“共识”来得到视频级的预测结果。在学习过程中，通过迭代更新模型参数来优化视频级预测的损失值（loss value）。其网络结构如下：

image 1

TSN_LSTM模型：

image 2



解压数据集，由于数据集过大，建议解压到data文件夹，防止每次打开项目花费过多时间
In [ ]
!mkdir data/dataset
!unzip data/data57932/20BN2.zip -d data/dataset
!unzip data/data57932/20BN1.zip -d data/dataset
In [ ]
%cd data/dataset/
/home/aistudio/data/dataset
In [ ]
#解压数据集
!cat 20bn-jester-v1-?? | tar zx
In [ ]
#删除原文件
!ls -i |grep "20bn-jester-v1-"|awk '{print $2}'|xargs -i rm -f {}
In [ ]
#运行两次回到/home/aistudio
%cd ..
%cd ..
/home/aistudio/data
/home/aistudio
数据读取
In [ ]
import pandas as pd
import numpy as np
label_dict={}
labels=pd.read_csv("data/dataset/jester-v1-labels.csv",header=None)
for index,label in enumerate(labels.values):
    label_dict[label[0]]=index
np.save('data/dataset/label_dir.npy', label_dict)
print(label_dict)    
{'Swiping Left': 0, 'Swiping Right': 1, 'Swiping Down': 2, 'Swiping Up': 3, 'Pushing Hand Away': 4, 'Pulling Hand In': 5, 'Sliding Two Fingers Left': 6, 'Sliding Two Fingers Right': 7, 'Sliding Two Fingers Down': 8, 'Sliding Two Fingers Up': 9, 'Pushing Two Fingers Away': 10, 'Pulling Two Fingers In': 11, 'Rolling Hand Forward': 12, 'Rolling Hand Backward': 13, 'Turning Hand Clockwise': 14, 'Turning Hand Counterclockwise': 15, 'Zooming In With Full Hand': 16, 'Zooming Out With Full Hand': 17, 'Zooming In With Two Fingers': 18, 'Zooming Out With Two Fingers': 19, 'Thumb Up': 20, 'Thumb Down': 21, 'Shaking Hand': 22, 'Stop Sign': 23, 'Drumming Fingers': 24, 'No gesture': 25, 'Doing other things': 26}
In [ ]
def generatetxt(csvpath,source_dir,target_dir):
    datas=pd.read_csv(csvpath,header=None)
    # file=open(savepath,  mode="w",encoding="utf8")
    for index,data in enumerate(datas.values):
        data=str(data[0]).split(';')
        if len(data)==1: #测试集
            image_file = os.listdir(os.path.join(source_dir, data[0]))
            image_file.sort()
            image_num = len(image_file)
            frame = []
            vid = data[0]
            for image_name in image_file:
                image_path = os.path.join(os.path.join(source_dir, data[0]), image_name)
                frame.append(image_path)
            output_pkl = vid + '.pkl'
            output_pkl = os.path.join(target_dir, output_pkl)
            f = open(output_pkl, 'wb')
            pickle.dump((vid, frame), f, -1)
            f.close()
        else:
            image_file = os.listdir(os.path.join(source_dir, data[0]))
            image_file.sort()
            image_num = len(image_file)
            frame = []
            vid = data[0]
            for image_name in image_file:
                image_path = os.path.join(os.path.join(source_dir, data[0]), image_name)
                frame.append(image_path)
            output_pkl = vid + '.pkl'
            output_pkl = os.path.join(target_dir, output_pkl)
            f = open(output_pkl, 'wb')
            pickle.dump((vid,label_dict[data[1]], frame), f, -1)
            f.close()
In [ ]
#生成训练，验证，测试集的list文件
import os
import numpy as np
import cv2
import sys
import glob
import pickle
from multiprocessing import Pool


source_dir = 'data/dataset/20bn-jester-v1'
train_dir = 'data/dataset/jester-v1-train.csv'
val_dir = 'data/dataset/jester-v1-validation.csv'
test_dir = 'data/dataset/jester-v1-test.csv'
target_train_dir = 'data/dataset/train'
target_test_dir = 'data/dataset/test'
target_val_dir = 'data/dataset/val'
if not os.path.exists(target_train_dir):
    os.mkdir(target_train_dir)
if not os.path.exists(target_test_dir):
    os.mkdir(target_test_dir)
if not os.path.exists(target_val_dir):
    os.mkdir(target_val_dir)

generatetxt(train_dir,source_dir,target_train_dir)
generatetxt(val_dir,source_dir,target_val_dir)
generatetxt(test_dir,source_dir,target_test_dir)
In [ ]
import os


data_dir = 'data/dataset/'

train_data = os.listdir(data_dir + 'train')
train_data = [x for x in train_data if not x.startswith('.')]
print(len(train_data))

模型训练
In [30]
!python train.py --use_gpu True --epoch 35 --pretrain checkpoints_models

模型评估
In [ ]
!python eval.py --weights 'checkpoints_models/tsn_lstm_model31' --use_gpu True

test_data = os.listdir(data_dir + 'test')
test_data = [x for x in test_data if not x.startswith('.')]
print(len(test_data))

val_data = os.listdir(data_dir + 'val')
val_data = [x for x in val_data if not x.startswith('.')]
print(len(val_data))

f = open('data/dataset/train.list', 'w')
for line in train_data:
    f.write(data_dir + 'train/' + line + '\n')
f.close()
f = open('data/dataset/test.list', 'w')
for line in test_data:
    f.write(data_dir + 'test/' + line + '\n')
f.close()
f = open('data/dataset/val.list', 'w')
for line in val_data:
    f.write(data_dir + 'val/' + line + '\n')
f.close()
118562
14743
14787


