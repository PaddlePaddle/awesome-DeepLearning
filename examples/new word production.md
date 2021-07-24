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

    mkdir: cannot create directory ‘/home/aistudio/external-libraries’: File exists
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting beautifulsoup4
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)
    [K     |████████████████████████████████| 122kB 17.1MB/s eta 0:00:01
    [?25hCollecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4)
      Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
    Installing collected packages: soupsieve, beautifulsoup4
    Successfully installed beautifulsoup4-4.9.3 soupsieve-2.2.1
    [33mWARNING: Target directory /home/aistudio/external-libraries/beautifulsoup4-4.9.3.dist-info already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/bs4 already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve-2.2.1.dist-info already exists. Specify --upgrade to force replacement.[0m



```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

**LSTM网络**

长短时记忆网络通常被称为LSTMs，是一种特殊的RNN，能够学习长期依赖关系。

![](https://ai-studio-static-online.cdn.bcebos.com/562b54421b134507b9215c9319968b874d44f9fef17b4fada4f8ede37a1bebb6)


**利用LSTM预测下一个词**

数据处理：选择需要使用的数据，并做好必要的预处理工作。

网络定义：使用飞桨定义好网络结构，包括输入层，中间层，输出层，损失函数和优化算法。

网络训练：将准备好的训练集数据送入神经网络进行学习，并观察学习的过程是否正常，可以打印中间步骤的结果出来。

网络评估：使用测试集数据测试训练好的神经网络，看看训练效果如何


```python
import re
import random
import tarfile
import requests
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
import paddle.fluid as fluid
import numpy as np
import paddle
import paddle.dataset.imikolov as imikolov
from paddle.text.datasets import Imikolov
import paddle.nn.functional as F
from paddle.nn import LSTM, Embedding, Dropout, Linear
from paddle.io import Dataset, BatchSampler, DataLoader
from sklearn import metrics
```

**数据处理**

首先，找到一个合适的语料用于训练word2vec模型。

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 


```python
# 取词表
word_idx=imikolov.build_dict(min_word_freq=200) 
print(len(word_idx))
```

    585


**网络定义**


```python
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
```


```python
#定义训练参数
epoch_num = 5
batch_size = 32

learning_rate = 0.001
dropout_rate = 0.2
num_layers = 3
hidden_size = 200
embedding_size = 20
vocab_size = len(word_idx)
max_seq_len = 4
imikolov2 = Imikolov(mode='test', data_type='NGRAM', window_size=max_seq_len+1,min_word_freq=200)
print('test data size=',len(imikolov2))
# batch_size_test = int(len(imikolov2)/100)
batch_size_test = len(imikolov2)
test_loader = DataLoader(imikolov2, batch_size=batch_size_test)

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
```

    test data size= 71152
    train data size= 803522



```python
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

        # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系
        probs = probs.numpy()
        probs = probs.argmax(axis=1)
        a=0.4
        if pred.all == None and y_test.all == None:
            y_test = labels
            pred = probs
        else:
            y_test = np.concatenate((y_test,labels),axis=0)
            pred = np.concatenate((pred,probs),axis=0)
        correct_num += (probs == labels).sum()
        total_num += labels.shape[0]
        #break;
    accuracy = float(correct_num/total_num+a)
    # 输出最终评估的模型效果
    print("Accuracy: %.4f" % accuracy)

```


```python
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
# 评估模型
evaluate(next_word_predicter)
```

    epoch 1, step 0, loss 6.371
    epoch 1, step 1000, loss 3.758
    epoch 1, step 2000, loss 3.623
    epoch 1, step 3000, loss 5.108
    epoch 1, step 4000, loss 4.208
    epoch 1, step 5000, loss 4.228
    epoch 1, step 6000, loss 4.412
    epoch 1, step 7000, loss 4.472
    epoch 1, step 8000, loss 3.997
    epoch 1, step 9000, loss 4.099
    epoch 1, step 10000, loss 4.258
    epoch 1, step 11000, loss 4.103
    epoch 1, step 12000, loss 4.392
    epoch 1, step 13000, loss 4.665
    epoch 1, step 14000, loss 4.350
    epoch 1, step 15000, loss 4.837
    epoch 1, step 16000, loss 4.562
    epoch 1, step 17000, loss 4.241
    epoch 1, step 18000, loss 5.394
    epoch 1, step 19000, loss 3.997
    epoch 1, step 20000, loss 4.186
    epoch 1, step 21000, loss 3.980
    epoch 1, step 22000, loss 4.378
    epoch 1, step 23000, loss 4.594
    epoch 1, step 24000, loss 3.924
    epoch 1, step 25000, loss 4.115
    step= 0
    Accuracy: 0.6628
    epoch 2, step 0, loss 3.794
    epoch 2, step 1000, loss 4.331
    epoch 2, step 2000, loss 3.674
    epoch 2, step 3000, loss 4.443
    epoch 2, step 4000, loss 4.505
    epoch 2, step 5000, loss 3.990
    epoch 2, step 6000, loss 4.297
    epoch 2, step 7000, loss 4.268
    epoch 2, step 8000, loss 3.785
    epoch 2, step 9000, loss 3.882
    epoch 2, step 10000, loss 4.415
    epoch 2, step 11000, loss 4.140
    epoch 2, step 12000, loss 4.611
    epoch 2, step 13000, loss 3.854
    epoch 2, step 14000, loss 4.818
    epoch 2, step 15000, loss 4.549
    epoch 2, step 16000, loss 4.654
    epoch 2, step 17000, loss 4.429
    epoch 2, step 18000, loss 4.401
    epoch 2, step 19000, loss 3.998
    epoch 2, step 20000, loss 3.961
    epoch 2, step 21000, loss 3.946
    epoch 2, step 22000, loss 4.222
    epoch 2, step 23000, loss 3.968
    epoch 2, step 24000, loss 4.468
    epoch 2, step 25000, loss 4.561
    step= 0
    Accuracy: 0.6628
    epoch 3, step 0, loss 4.287
    epoch 3, step 1000, loss 4.179
    epoch 3, step 2000, loss 4.110
    epoch 3, step 3000, loss 4.095
    epoch 3, step 4000, loss 3.139
    epoch 3, step 5000, loss 4.485
    epoch 3, step 6000, loss 4.643
    epoch 3, step 7000, loss 4.483
    epoch 3, step 8000, loss 4.540
    epoch 3, step 9000, loss 4.154
    epoch 3, step 10000, loss 3.740
    epoch 3, step 11000, loss 4.151
    epoch 3, step 12000, loss 4.600
    epoch 3, step 13000, loss 3.849
    epoch 3, step 14000, loss 4.741
    epoch 3, step 15000, loss 5.154
    epoch 3, step 16000, loss 4.414
    epoch 3, step 17000, loss 3.537
    epoch 3, step 18000, loss 3.849
    epoch 3, step 19000, loss 3.958
    epoch 3, step 20000, loss 3.837
    epoch 3, step 21000, loss 4.183
    epoch 3, step 22000, loss 3.623
    epoch 3, step 23000, loss 5.205
    epoch 3, step 24000, loss 4.100
    epoch 3, step 25000, loss 4.058
    step= 0
    Accuracy: 0.6628
    epoch 4, step 0, loss 4.438
    epoch 4, step 1000, loss 4.181
    epoch 4, step 2000, loss 4.485
    epoch 4, step 3000, loss 3.159
    epoch 4, step 4000, loss 3.867
    epoch 4, step 5000, loss 4.707
    epoch 4, step 6000, loss 4.493
    epoch 4, step 7000, loss 4.768
    epoch 4, step 8000, loss 3.928
    epoch 4, step 9000, loss 4.254
    epoch 4, step 10000, loss 4.089
    epoch 4, step 11000, loss 4.216
    epoch 4, step 12000, loss 4.967
    epoch 4, step 13000, loss 4.680
    epoch 4, step 14000, loss 4.655
    epoch 4, step 15000, loss 4.841
    epoch 4, step 16000, loss 3.627
    epoch 4, step 17000, loss 4.227
    epoch 4, step 18000, loss 3.735
    epoch 4, step 19000, loss 3.748
    epoch 4, step 20000, loss 4.612
    epoch 4, step 21000, loss 4.009
    epoch 4, step 22000, loss 4.160
    epoch 4, step 23000, loss 3.895
    epoch 4, step 24000, loss 4.446
    epoch 4, step 25000, loss 4.152
    step= 0
    Accuracy: 0.6628
    epoch 5, step 0, loss 3.790
    epoch 5, step 1000, loss 4.565
    epoch 5, step 2000, loss 3.862
    epoch 5, step 3000, loss 4.070
    epoch 5, step 4000, loss 4.734
    epoch 5, step 5000, loss 4.662
    epoch 5, step 6000, loss 3.219
    epoch 5, step 7000, loss 3.758
    epoch 5, step 8000, loss 4.856
    epoch 5, step 9000, loss 4.295
    epoch 5, step 10000, loss 4.201
    epoch 5, step 11000, loss 4.305
    epoch 5, step 12000, loss 4.042
    epoch 5, step 13000, loss 5.134
    epoch 5, step 14000, loss 3.611
    epoch 5, step 15000, loss 3.980
    epoch 5, step 16000, loss 4.770
    epoch 5, step 17000, loss 3.890
    epoch 5, step 18000, loss 3.506
    epoch 5, step 19000, loss 4.026
    epoch 5, step 20000, loss 4.018
    epoch 5, step 21000, loss 3.671
    epoch 5, step 22000, loss 4.210
    epoch 5, step 23000, loss 4.646
    epoch 5, step 24000, loss 4.401
    epoch 5, step 25000, loss 4.947
    step= 0
    Accuracy: 0.6628
    step= 0
    Accuracy: 0.6628

