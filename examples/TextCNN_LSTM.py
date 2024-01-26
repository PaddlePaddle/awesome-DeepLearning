#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# **TextCNN**
# **TextCNN模型结构**
# ![](https://ai-studio-static-online.cdn.bcebos.com/72c86bfa5a1f4afdac7f544b317dca5088e6755541534f93a48c226daa8acbd9)
# ![](https://ai-studio-static-online.cdn.bcebos.com/65c8d8a456834f9f8d06b032a5b7cedc120403030e30401ab3bc8c82fd7ad631)
# **嵌入层(embedding layer)**
# 	TextCNN使用预先训练好的词向量作嵌入层。对于数据集里的所有词，因为每个词都可以表征成一个向量，因此可以得到一个嵌入矩阵MM, MM里的每一行都是词向量。这个MM可以是静态(static)的，也就是固定不变。可以是非静态(non-static)的，也就是可以根据反向传播更新。
# 	多种模型：
# 	CNN-rand：作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。
# 	CNN-static：模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。
# 	CNN-non-static：同（2），只是训练的时候，Embedding layer跟随整个网络一起训练。
# 	CNN-multichannel：Embedding layer有两个channel，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。
# **卷积池化层(convolution and pooling)**
# **卷积(convolution)**
# 	输入一个句子，首先对这个句子进行切词，假设有s个单词。对每个词，跟句嵌入矩阵M, 可以得到词向量。假设词向量一共有d维。那么对于这个句子，便可以得到s行d列的矩阵AϵRs×d. 
# 	我们可以把矩阵A看成是一幅图像，使用卷积神经网络去提取特征。由于句子中相邻的单词关联性总是很高的，因此可以使用一维卷积，即文本卷积与图像卷积的不同之处在于只在文本序列的一个方向（垂直）做卷积，卷积核的宽度固定为词向量的维度d。高度是超参数，可以设置。 对句子单词每个可能的窗口做卷积操作得到特征图(feature map) c = [c_1, c_2, …, c_s-h+1]。
# 	对一个卷积核，可以得到特征cϵRs−h+1, 总共s−h+1个特征。我们可以使用更多高度h不同的卷积核，得到更丰富的特征表达。
# **池化(pooling)**
# 	不同尺寸的卷积核得到的特征(feature map)大小也是不一样的，因此我们对每个feature map使用池化函数，使它们的维度相同。
# 	Max Pooling
# 		最常用的就是1-max pooling，提取出feature map照片那个的最大值，通过选择每个feature map的最大值，可捕获其最重要的特征。这样每一个卷积核得到特征就是一个值，对所有卷积核使用1-max pooling，再级联起来，可以得到最终的特征向量，这个特征向量再输入softmax layer做分类。这个地方可以使用drop out防止过拟合。
# ![](https://ai-studio-static-online.cdn.bcebos.com/0a282adc13ea4b7ead6d0512e5b1d5d8af798e6344a247e7928f117fde111a57)
# 		CNN中采用Max Pooling操作有几个好处：首先，这个操作可以保证特征的位置与旋转不变性，因为不论这个强特征在哪个位置出现，都会不考虑其出现位置而能把它提出来。但是对于NLP来说，这个特性其实并不一定是好事，因为在很多NLP的应用场合，特征的出现位置信息是很重要的，比如主语出现位置一般在句子头，宾语一般出现在句子尾等等。     其次，MaxPooling能减少模型参数数量，有利于减少模型过拟合问题。因为经过Pooling操作后，往往把2D或者1D的数组转换为单一数值，这样对于后续的Convolution层或者全联接隐层来说无疑单个Filter的参数或者隐层神经元个数就减少了。 再者，对于NLP任务来说，可以把变长的输入X整理成固定长度的输入。因为CNN最后往往会接全联接层，而其神经元个数是需要事先定好的，如果输入是不定长的那么很难设计网络结构。
# 
# 		但是，CNN模型采取MaxPooling Over Time也有缺点：首先特征的位置信息在这一步骤完全丢失。在卷积层其实是保留了特征的位置信息的，但是通过取唯一的最大值，现在在Pooling层只知道这个最大值是多少，但是其出现位置信息并没有保留；另外一个明显的缺点是：有时候有些强特征会出现多次，出现次数越多说明这个特征越强，但是因为Max Pooling只保留一个最大值，就是说同一特征的强度信息丢失了。
# 	K-Max Pooling
# 		取所有特征值中得分在Top –K的值，并（保序拼接）保留这些特征值原始的先后顺序（即多保留一些特征信息供后续阶段使用）。[A Convolutional Neural Network for Modelling Sentences]。
# 		![](https://ai-studio-static-online.cdn.bcebos.com/1aeb3d2e64164978b88a15dd393ef1f7fd8da1b1798742129f9a462fa15358fc)
#         
# 

# In[1]:


import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random
 
USE_CUDA = torch.cuda.is_available()
 
# 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)
 
BATCH_SIZE = 32 #一个BATCH里有多少个句子
EMBEDDING_SIZE = 650 #输入的时候把单词embed成多少维
MAX_VOCAB_SIZE = 50000

TEXT = torchtext.data.Field(lower=True)
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=".", 
    train="text8.train.txt", #训练集
    validation="text8.dev.txt", #验证集
    test="text8.test.txt", #测试集
    text_field=TEXT)
TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
print("vocabulary size: {}".format(len(TEXT.vocab)))
 
device = torch.device("cuda" if USE_CUDA else "cpu")
VOCAB_SIZE = len(TEXT.vocab)
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, val, test), 
    batch_size=BATCH_SIZE, 
    device=device, 
    bptt_len=32,
    repeat=False, #使epoch为1
    shuffle=True
)

it = iter(train_iter)
batch = next(it)
print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:,1].data]))
print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,1].data]))

import torch
import torch.nn as nn
 
class RNNModel(nn.Module):
    """ 一个简单的循环神经网络
    @ntoken是vocabulary size
    @ninp是embed_size
    """
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        ''' 该模型包含以下几层:
            - 词嵌入层
            - 一个循环神经网络层(RNN, LSTM, GRU)
            - 一个线性层，从hidden state到输出单词表
            - 一个dropout层，用来做regularization
        '''
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
 
        self.init_weights()
 
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
 
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
 
    def forward(self, input, hidden):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden
 
    def init_hidden(self, bsz, requires_grad=True):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad),
                    weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad))
        else:
            return weight.new_zeros((self.nlayers, bsz, self.nhid), requires_grad=requires_grad)

