

import re
import nltk
# nltk.download('punkt')
import paddle
import numpy as np
import paddle.nn as nn
import paddle.fluid as fluid
import matplotlib.pyplot as plt
from sklearn import preprocessing  
from nltk.tokenize import word_tokenize


# 将字母全部转为小写，并按照
def preprocess(corpus):
    corpus = corpus.strip().lower()
    corpus=re.sub('[^a-z0-9]+',' ', corpus)
    corpus = word_tokenize(corpus)
    return corpus

def build_dict(corpus):
    word_freq = dict()
    for word in corpus:
        if word not in word_freq:
            word_freq[word] = 0
        word_freq[word] += 1

    word_freq = sorted(word_freq.items(), key = lambda x:x[1], reverse = True)

    #构造3个不同的词典，分别存储，
    #每个词到id的映射关系：word2id_dict
    #每个id出现的频率：word2id_freq
    #每个id到词典映射关系：id2word_dict
    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    #按照频率，从高到低，开始遍历每个单词，并为这个单词构造id
    for word, freq in word_freq:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict

#将语料库中的单词装换为对应的id
def convert_corpus_to_id(corpus, word2id_dict):
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

# 构建模型
class myLSTM(nn.Layer):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(myLSTM, self).__init__()

        # num_embeddings (int) - 嵌入字典的大小， input中的id必须满足 0 =< id < num_embeddings 。 。
        # embedding_dim (int) - 每个嵌入向量的维度。
        # padding_idx (int|long|None) - padding_idx的配置区间为 [-weight.shape[0], weight.shape[0]，如果配置了padding_idx，那么在训练过程中遇到此id时会被用
        # sparse (bool) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
        # weight_attr (ParamAttr|None) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 ParamAttr ，一般无需设置，默认值为None。
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # input_size (int) - 输入的大小。
        # hidden_size (int) - 隐藏状态大小。
        # num_layers (int，可选) - 网络层数。默认为1。
        # direction (str，可选) - 网络迭代方向，可设置为forward或bidirect（或bidirectional）。默认为forward。
        # time_major (bool，可选) - 指定input的第一个维度是否是time steps。默认为False。
        # dropout (float，可选) - dropout概率，指的是出第一层外每层输入时的dropout概率。默认为0。
        # weight_ih_attr (ParamAttr，可选) - weight_ih的参数。默认为None。
        # weight_hh_attr (ParamAttr，可选) - weight_hh的参数。默认为None。
        # bias_ih_attr (ParamAttr，可选) - bias_ih的参数。默认为None。
        # bias_hh_attr (ParamAttr，可选) - bias_hh的参数。默认为None。
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2)

        # in_features (int) – 线性变换层输入单元的数目。
        # out_features (int) – 线性变换层输出单元的数目。
        # weight_attr (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 ParamAttr 。
        # bias_attr (ParamAttr|bool, 可选) – 指定偏置参数的属性。 bias_attr 为bool类型且设置为False时，表示不会为该层添加偏置。 bias_attr 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 ParamAttr 。默认值为None。
        # name (str，可选) – 具体用法请参见 Name ，一般无需设置，默认值为None。
        self.linear = nn.Linear(hidden_size*seq_length, out_features=vocab_size,bias_attr=True)

    

    def forward(self, input_word):
        
        emb = self.embedding(input_word)
        
        output, hidden = self.lstm(emb)

        # output = output.view(output.size(0), -1)
        output =paddle.reshape(output,(output.shape[0], -1))

        
        output = self.linear(output)

        return output, hidden

corpus=open("./data/data101239/corpus.txt").read()
corpus = preprocess(corpus)

word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)

corpus = convert_corpus_to_id(corpus, word2id_dict)

#length of the sequence to train
train_len = 3

text_sequences = []
for i in range(train_len,len(corpus)+1):
  seq = corpus[i-train_len:i]
  text_sequences.append(seq)

sequences=np.asarray(text_sequences)

#vocabulary size
vocabulary_size = len(word2id_dict)+1

#trainX
train_inputs=sequences[:,:-1]

#input sequence length
seq_length=train_inputs.shape[1]

#trainY
train_targets=sequences[:,-1]

one_hot_label = np.zeros(shape=(train_targets.shape[0],vocabulary_size)) #生成全0矩阵
one_hot_label[np.arange(0,train_targets.shape[0]),train_targets] = 1 #相应标签位置置1

model=myLSTM(vocab_size=vocabulary_size,embed_size=128, hidden_size=256)

#Adam optimizer
optimizer= paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.07)

#loss
criterion = nn.BCEWithLogitsLoss()

#number of epoch
no_epoch=50
losses=[]
accuracy = []

with fluid.dygraph.guard(paddle.CUDAPlace(0)):
    
    for epoch in range(1,no_epoch+1):
        model.train()
        tr_loss = 0

        y_pred, (state_h, state_c) = model(paddle.to_tensor(train_inputs))

        loss = criterion(y_pred, paddle.to_tensor(one_hot_label.astype('float32')))
        losses.append(loss.numpy().item())

        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        model.eval()

        y_pred, (state_h, state_c) = model(paddle.to_tensor(train_inputs))

        acc = (np.argmax(y_pred.numpy(),axis=1) == train_targets).sum() / len(train_targets) 
        accuracy.append(acc)

        print("Epoch : ",epoch,"loss : ",loss.numpy().item(), 'accuracy : ',acc)

#plotting the loss, loss is decreasing for each epoch
plt.plot(losses, label='Training loss')
plt.legend(loc="upper right")   #显示图中的标签
plt.show()

plt.plot(accuracy, label='Accuracy')
plt.legend(loc="upper left")   #显示图中的标签
plt.show()

