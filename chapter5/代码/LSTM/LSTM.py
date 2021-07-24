#!/usr/bin/env python
# coding: utf-8

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# import

# In[ ]:


import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.dataset.imikolov as imikolov# 导入imikolov数据集
from paddle.text.datasets import Imikolov


# In[ ]:


import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
from paddle.io import Dataset, BatchSampler, DataLoader
from sklearn import metrics


# In[ ]:


# 取词表
word_idx=imikolov.build_dict(min_word_freq=200) #min_word_freq=50
print(len(word_idx))


# In[ ]:


class PredictNextWord(paddle.nn.Layer):# 预测下一个词
    def __init__(self, hidden_size, vocab_size, embedding_size, class_num, num_steps=4, num_layers=1, init_scale=0.1, dropout_rate=None):
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.embedding_size，表示词向量的维度
        # 4.class_num，分类个数，等同于vocab_size
        # 5.num_steps，表示模型最大可以考虑的句子长度
        # 6.num_layers，表示网络的层数
        # 7.dropout_rate，表示使用dropout过程中失活的神经元比例
        # 8.init_scale，表示网络内部的参数的初始化范围
        
        super(PredictNextWord, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale

        # embedding 将词转化为词向量
        self.embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, sparse=False, 
                                    weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))

        # 构建LSTM模型
        self.simple_lstm_rnn = paddle.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)
        self.cls_fc = paddle.nn.Linear(in_features=self.num_steps*self.hidden_size, out_features=self.class_num)
        # dropout
        self.dropout_layer = paddle.nn.Dropout(p=self.dropout_rate, mode='upscale_in_train')


    # forwad函数为模型前向计算的函数
    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # 定义LSTM的初始hidden和cell
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_hidden = paddle.to_tensor(init_hidden_data)#hidden
        init_cell = paddle.to_tensor(init_cell_data)#cell

        x_emb = self.embedding(inputs)
        x_emb = paddle.reshape(x_emb, shape=[-1, self.num_steps, self.embedding_size])#get embedding

        # dropout
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x_emb = self.dropout_layer(x_emb)


        # 使用LSTM网络，把每个句子转换为语义向量
        rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb, (init_hidden, init_cell))
        #rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb)
        # 提取最后一层隐状态作为文本的语义向量
        rnn_out = paddle.reshape(rnn_out, shape=[batch_size, -1])
        # 将每个句子的向量表示映射到具体的类别上, logits的维度为[batch_size, vocab_size]
        logits = self.cls_fc(rnn_out)
        return logits


# In[ ]:


max_seq_len = 4
imikolov2 = Imikolov(mode='test', data_type='NGRAM', window_size=max_seq_len+1,min_word_freq=200)
print('test data size=',len(imikolov2))
# batch_size_test = int(len(imikolov2)/100)
batch_size_test = len(imikolov2)
test_loader = DataLoader(imikolov2, batch_size=batch_size_test)


# In[ ]:


def evaluate(model):# 测试
    model.eval()
    correct_num = 0
    total_num = 0
    y_test = np.array([])
    pred = np.array([])
    for step, data in enumerate(test_loader()):
        print('step=',step)
        data = np.array(data)
        # print(data.shape)
        if data.shape[1] < batch_size_test:
                break
        else:
            data = data.reshape(batch_size_test,-1)
        sentences = data[:,:4]
        labels = data[:,-1]
        # 将张量转换为Tensor类型
        sentences = paddle.to_tensor(sentences)
        labels = paddle.to_tensor(labels)
        logits = model(sentences)
        labels = labels.numpy()


        probs = F.softmax(logits)
        probs = probs.numpy()
        probs = probs.argmax(axis=1)
        if pred.all == None and y_test.all == None:
            y_test = labels
            pred = probs
        else:
            y_test = np.concatenate((y_test,labels),axis=0)
            pred = np.concatenate((pred,probs),axis=0)
        correct_num += (probs == labels).sum()
        total_num += labels.shape[0]
    accuracy = float(correct_num/total_num)

    print("Accuracy: %.4f" % accuracy)
    print('y_test=', y_test)
    print('pred=', pred)
    accuracy = metrics.accuracy_score(y_test, pred)
    overall_precison = metrics.precision_score(y_test, pred, average="micro")
    average_precison = metrics.precision_score(y_test, pred, average="macro")
    overall_recall = metrics.recall_score(y_test, pred, average="micro")
    average_recall = metrics.recall_score(y_test, pred, average="macro")
    print('accuracy = ', accuracy)
    print('overall_precison = ', overall_precison)
    print('average_precison = ', average_precison)
    print('overall_recall = ', overall_recall)
    print('average_recall = ', average_recall)


# In[11]:


# 定义训练参数
epoch_num = 10
batch_size = 32
learning_rate = 0.02
dropout_rate = 0.2
num_layers = 3
hidden_size = 200
embedding_size = 20
vocab_size = len(word_idx)
# 数据生成器
imikolov = Imikolov(mode='train', data_type='NGRAM', window_size=max_seq_len+1,min_word_freq=200)
print('train data size=',len(imikolov))
train_loader = DataLoader(imikolov, batch_size=batch_size, shuffle=True)

# 使用GPU
paddle.set_device('gpu:0')

next_word_predicter = PredictNextWord(hidden_size, vocab_size, embedding_size, class_num=vocab_size, num_steps=max_seq_len, num_layers=num_layers, dropout_rate=dropout_rate)

optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, parameters= next_word_predicter.parameters()) # , beta1=0.9, beta2=0.999,

# 定义训练函数
losses = []
steps = []
def train(model):# TRAIN
    for e in range(epoch_num):
        model.train()
        for step, data in enumerate(train_loader()):
            data = np.array(data)
            if data.shape[1] < batch_size:
                break
            else:
                data = data.reshape(batch_size,-1)

            sentences = data[:,:4]
            labels = data[:,-1]
            sentences = paddle.to_tensor(sentences)
            labels = paddle.to_tensor(labels)
            # 前向计算，将数据feed进模型，并得到预测的情感标签和损失
            logits = model(sentences)
            # logits = F.softmax(logits)
            # 计算损失
            loss = F.cross_entropy(input=logits, label=labels, soft_label=False)
            loss = paddle.mean(loss)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if step % 1000 == 0:
                losses.append(loss.numpy()[0])
                steps.append(step)
                # 打印当前loss数值
                print("epoch %d, step %d, loss %.3f" % (e+1, step, loss.numpy()[0]))
        evaluate(model)


# In[10]:


train(next_word_predicter)


# In[ ]:




