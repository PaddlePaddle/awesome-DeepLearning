
###基于层次softmax训练词向量
假设我们vocabulary size 为V，hidden layer 神经元个数为N，假设我们只有一个上下文单词，则根据这个上下文单词预测目标词，类似于一个bigram model，如下图所示
![Alt text](./1.1.png)
输入是一个one-hot编码的vector（大小为V），假设只给定一个上下文word，对于输入编码， { x 1 , x 2 , . . . , x v }，只有一个为1，其它都为0。
如上图所示，第一层的参数权重 W V ∗ N 
W中的每一行是一个N维度的向量，代表的就是单词 w的向量Vm表示。
从hidden layer到output layer，也有一个不同的权重矩阵W’={w’ij},是一个N*V的矩阵，第j列代表了词Wj的N维度向量，用这个向量和hidden layer输出向量相乘，就得到在中的每个词的分值uj
![Alt text](./1.2.png)
然后用softmax一个log 线性分类器，得到每个词的分布概率
![Alt text](./1.3.png)
其中wi是上下文单词，wj是目标词，yj是output layer层的第j个神经元的输出。Vw和V‘w是词W的两个词向量表示，Vw来自input–>hidden的权重矩阵W的行，而V‘w来自hidden—>output的权重矩阵W’的列，通过上面的语言模型，第一层权重W就是我们学习的词向量矩阵。
![Alt text](./1.4.png)
从公式（2）和（5）可知，更新最后一层的权重梯度， 我们必须计算词典V中的每个词，当 V很大的时候，最后一层softmax计算量会非常的耗时。

####hierarchical softmax
从上面的公式（2）可以看出，softmax分母那项归一化，每次需要计算所有的V的输出值，才可以得到当前j节点的输出，当V很大的时候，O（V）计算代价会非常高。所以在训练word2vec模型的时候，用到了两个tricks，一个是negative sampling，每次只采少量的负样本，不需要计算全部的V，另外一个是hierarchical softmax，通过构建赫夫曼tree来做层级softmax，从复杂度O（V）降低到O（log2V）。
####赫夫曼tree
![Alt text](./1.5.png)
首先对所有在V词表的词，根据词频来构建赫夫曼tree，词频越大，路径越短，编码信息更少。tree中的所有的叶子节点构成了词V，中间节点则共有V-1个，上面的每个叶子节点存在唯一的从根到该节点的path，如上图所示，词w2的path n（w2，1）、n（w2，2）、n（w2，3），其中n（w，j）表示词w的path的第j个节点	。
####怎么训练hierarchical softmax
#####预处理：构建haffman树
根据语料中的每个word的词频构建赫夫曼tree，词频越高，则离树根越近，路径越短。如上图所示，词典 V中的每个word都在叶子节点上，每个word需要计算两个信息：路径（经过的每个中间节点）以及赫夫曼编码，例如：
“respond”的路径经过的节点是（6，4，3），编码label是（1，0，0）
构建完赫夫曼tree后，每个叶子节点都有唯一的路径和编码，hierarchical softmax与softmax不同的是，在hierarchical softmax中，不对 V V V中的word词进行向量学习，而是对中间节点进行向量学习，而每个叶子上的节点可以通过路径中经过的中间节点去表示。

#####模型的输入
输入部分，在cbow或者skip-gram模型，要么是上下文word对应的id词向量平均，要么是中心词对应的id向量，作为hidden层的输出向量

#####样本label
不同softmax的是，每个词word对应的是一个V大小的one-hot label，hierarchical softmax中每个叶子节点word，对应的label是赫夫曼编码，一般长度不超过log2V，在训练的时候，每个叶子节点的label统一编码到一个固定的长度，不足的可以进行pad。

#####训练过程
我们用一个例子来描述，假如一个训练样本如下：
![Alt text](./1.6.png)
![Alt text](./1.7.png)
假如我们用skip-gram模型，则第一部分，根据"chupacabra" 词的one-hot编码乘以W权重矩阵，得到“chupachabra”的词向量表示，也就是hidden的输出，根据目标词“active”从赫夫曼tree中得到它的路径path，即经过的节点（6，4，3），而这些中间节点的向量是模型参数需要学习的，共有V-1个向量，通过对应的节点id，取出相应的向量，假设是w‘（3*N），分别与hidden输出向量相乘，再经过sigmoid函数，得到一个3*1的score分值，与实际label [1,0,1], 通过如下公式：
![Alt text](./1.8.png)
计算出loss，则就可以用优化器来计算模型中各参数的梯度进行更新了。



###LSTM实现词性标注

![Alt text](./LSTM.png)
把词向量 w 放入lstm 中训练，根据对应得tags 计算loss 并且跟新模型权重。
载入POS tagging训练和dev数据集。这些文件都是tab分隔的text和POS tag数据。
定义词性标注模型，

```
class POSTagger(nn.Module):
    def __init__(self, rnn_class, embedding_dim, hidden_dim, vocab_size, target_size, num_layers):
        super(POSTagger, self).__init__()

        self.embed = nn.Embedding(vocab_size,embedding_dim)
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

        self.rnn_type = rnn_class
        self.nhid = hidden_dim
        self.nlayers = num_layers
        self.rnn = getattr(nn, self.rnn_type)(embedding_dim, self.nhid, self.nlayers,bidirectional=True, dropout=0.5)
        self.output = nn.Linear(2*self.nhid,tag_vocab_size)

        self.drop = nn.Dropout(0.5)

    def forward(self, sentences):

        inputs = self.embed(sentences.long().cuda())

        x_emb = self.drop(inputs)

        hidden,states = self.rnn(x_emb)

        tag_scores = self.output(hidden)
        return tag_scores
```

```
model = POSTagger("LSTM", EMBEDDING_DIM, HIDDEN_DIM, text_vocab_size, tag_vocab_size, 2)
if USE_CUDA:
    model = model.cuda()
```

```
LR = 0.001
GAMMA = 1.
STEP_SIZE = 10
NUM_EPOCHS = 10
SAVE_DIR = "./save/"
loss_fn = nn.CrossEntropyLoss(size_average=False)
optimizer = optim.Adam(model.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
model = train_model(model, dataloaders, loss_fn, optimizer, exp_lr_scheduler, SAVE_DIR, NUM_EPOCHS, use_gpu=USE_CUDA)
```


