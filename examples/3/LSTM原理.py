#!/usr/bin/env python
# coding: utf-8

# In[5]:


import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.dataset.imikolov as imikolov
from paddle.text.datasets import Imikolov
import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
from paddle.io import Dataset, BatchSampler, DataLoader
from sklearn import metrics


# In[6]:


# 取词表
word_idx=imikolov.build_dict(min_word_freq=200) #min_word_freq=50
print(len(word_idx))

class NextWordPredicter(paddle.nn.Layer):

    def __init__(self, hidden_size, vocab_size, embedding_size, class_num, num_steps=4, num_layers=1, init_scale=0.1, dropout_rate=None):

        # 参数含义如下：
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.embedding_size，表示词向量的维度
        # 4.class_num，分类个数，等同于vocab_size
        # 5.num_steps，表示模型最大可以考虑的句子长度
        # 6.num_layers，表示网络的层数
        # 7.dropout_rate，表示使用dropout过程中失活的神经元比例
        # 8.init_scale，表示网络内部的参数的初始化范围,长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数，\
        # 这些函数对数值精度非常敏感，因此我们一般只使用比较小的初始化范围，以保证效果
        super(NextWordPredicter, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale

        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, sparse=False, 
                                    weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))
        # self.embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        # 声明一个LSTM模型，用来把每个句子抽象成向量
        self.simple_lstm_rnn = paddle.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers)

        # 声明使用上述语义向量映射到具体情感类别时所需要使用的线性层
        # self.cls_fc = paddle.nn.Linear(in_features=self.num_steps*self.hidden_size, out_features=self.class_num, 
                             # weight_attr=None, bias_attr=None)
        self.cls_fc = paddle.nn.Linear(in_features=self.num_steps*self.hidden_size, out_features=self.class_num)

        # 一般在获取单词的embedding后，会使用dropout层，防止过拟合，提升模型泛化能力
        self.dropout_layer = paddle.nn.Dropout(p=self.dropout_rate, mode='upscale_in_train')

    # forwad函数即为模型前向计算的函数，它有两个输入，分别为：
    # input为输入的训练文本，其shape为[batch_size, max_seq_len]
    # label训练文本对应的下一个词标签，其shape维[batch_size, 1]
    def forward(self, inputs):
        # 获取输入数据的batch_size
        batch_size = inputs.shape[0]

        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')
        init_cell_data = np.zeros(
            (self.num_layers, batch_size, self.hidden_size), dtype='float32')

        init_hidden = paddle.to_tensor(init_hidden_data)
        #init_hidden.stop_gradient = True
        init_cell = paddle.to_tensor(init_cell_data)
        #init_cell.stop_gradient = True

        # 将输入的句子的mini-batch转换为词向量表示，转换后输入数据shape为[batch_size, max_seq_len, embedding_size]
        x_emb = self.embedding(inputs)
        x_emb = paddle.reshape(x_emb, shape=[-1, self.num_steps, self.embedding_size])
        # 在获取的词向量后添加dropout层
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x_emb = self.dropout_layer(x_emb)

        # 使用LSTM网络，把每个句子转换为语义向量
        # 返回的rnn_out即为最后一个时间步的输出
        rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb, (init_hidden, init_cell))
        #rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb)
        # 提取最后一层隐状态作为文本的语义向量
        rnn_out = paddle.reshape(rnn_out, shape=[batch_size, -1])

        # 将每个句子的向量表示映射到具体的类别上, logits的维度为[batch_size, vocab_size]
        logits = self.cls_fc(rnn_out)
        return logits


# In[7]:


max_seq_len = 4
imikolov2 = Imikolov(mode='test', data_type='NGRAM', window_size=max_seq_len+1,min_word_freq=200)
print('test data size=',len(imikolov2))
# batch_size_test = int(len(imikolov2)/100)
batch_size_test = len(imikolov2)
test_loader = DataLoader(imikolov2, batch_size=batch_size_test)


# In[8]:


def evaluate(model):
    # 开启模型测试模式，在该模式下，网络不会进行梯度更新
    model.eval()

    # 构造测试数据生成器
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

        # 获取模型对当前batch的输出结果
        logits = model(sentences)
        labels = labels.numpy()
        # 使用softmax进行归一化
        probs = F.softmax(logits)

        # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
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
        #break;


    # 整体准确率
    # accuracy = (tp + tn) / (tp + tn + fp + fn)
    accuracy = float(correct_num/total_num)
    # 输出最终评估的模型效果
    # print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
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


# In[ ]:


# 定义训练参数
epoch_num = 5
batch_size = 32

learning_rate = 0.01
dropout_rate = 0.2
num_layers = 3
hidden_size = 200
embedding_size = 20
vocab_size = len(word_idx)

# 数据生成器
imikolov = Imikolov(mode='train', data_type='NGRAM', window_size=max_seq_len+1,min_word_freq=200)
print('train data size=',len(imikolov))
train_loader = DataLoader(imikolov, batch_size=batch_size, shuffle=True)

# 检测是否可以使用GPU，如果可以优先使用GPU
use_gpu = True if paddle.get_device().startswith("gpu") else False
if use_gpu:
    paddle.set_device('gpu:0')

# 实例化模型
next_word_predicter = NextWordPredicter(hidden_size, vocab_size, embedding_size, class_num=vocab_size, num_steps=max_seq_len, num_layers=num_layers, dropout_rate=dropout_rate)

# 指定优化策略，更新模型参数
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, parameters= next_word_predicter.parameters()) # , beta1=0.9, beta2=0.999,
# optimizer = paddle.optimizer.SGD(learning_rate=learning_rate,parameters= next_word_predicter.parameters())
# 定义训练函数
# 记录训练过程中的损失变化情况，可用于后续画图查看训练情况
losses = []
steps = []


# In[ ]:


def train(model):
    # 开启模型训练模式

    # 建立训练数据生成器，每次迭代生成一个batch，每个batch包含训练文本和文本对应的情感标签
    for e in range(epoch_num):
        model.train()
        for step, data in enumerate(train_loader()):
            data = np.array(data)
            if data.shape[1] < batch_size:
                break
            else:
                data = data.reshape(batch_size,-1)
            # 获取数据，并将张量转换为Tensor类型
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

            # 后向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()

            if step % 1000 == 0:
                # 记录当前步骤的loss变化情况
                losses.append(loss.numpy()[0])
                steps.append(step)
                # 打印当前loss数值
                print("epoch %d, step %d, loss %.3f" % (e+1, step, loss.numpy()[0]))
                # print('label=',labels)
                # print('predict=',logits.argmax(axis=1))
        evaluate(model)


# In[ ]:


#训练模型
train(next_word_predicter)
# 保存模型，包含两部分：模型参数和优化器参数
model_name = "next_word_predicter"
# 保存训练好的模型参数
paddle.save(next_word_predicter.state_dict(), "{}.pdparams".format(model_name))
# 保存优化器参数，方便后续模型继续训练
paddle.save(optimizer.state_dict(), "{}.pdopt".format(model_name))

# 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
saved_state = paddle.load("./next_word_predicter.pdparams")
next_word_predicter = NextWordPredicter(hidden_size, vocab_size, embedding_size,class_num=vocab_size, num_steps=max_seq_len, num_layers=num_layers, dropout_rate=dropout_rate)
next_word_predicter.load_dict(saved_state)

def predict(model,index):
    data = imikolov2[index]
    data = np.array(data)
    real = [word_idx_convert[i] for i in data]
    print('real: ', real)
    sentences = data[:4]
    predict = sentences.copy()
    sentences = np.expand_dims(sentences, 0)
    sentences = paddle.to_tensor(sentences)
    logits = model(sentences)
    logits = logits.argmax(axis=1)
    predict = np.concatenate((predict,logits),axis=0)
    predict_s = [word_idx_convert[i] for i in predict]
    print('predict:', predict_s)

# 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
word_idx_convert = dict([(v,k) for (k,v) in word_idx.items()])
saved_state = paddle.load("./next_word_predicter.pdparams")
next_word_predicter = NextWordPredicter(hidden_size, vocab_size, embedding_size,class_num=vocab_size, num_steps=max_seq_len, num_layers=num_layers, dropout_rate=dropout_rate)
next_word_predicter.load_dict(saved_state)

predict(next_word_predicter, 60)

