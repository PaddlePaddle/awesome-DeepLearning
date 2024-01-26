#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


all_data = pd.read_csv("data/data69671/all_data.tsv", sep="\t")
all_data.head()


# ### 生成词典

# In[3]:



all_str = all_data["text"].values.tolist()
dict_set = set() # 保证每个字符只有唯一的对应数字
for content in all_str:
    for s in content:
        dict_set.add(s)
# 添加未知字符
dict_set.add("<unk>")
# 把元组转换成字典，一个字对应一个数字
dict_list = []
i = 0
for s in dict_set:
    dict_list.append([s, i])
    i += 1
dict_txt = dict(dict_list)
# 字典保存到本地
with open("dict.txt", 'w', encoding='utf-8') as f:
    f.write(str(dict_txt))


# In[4]:


# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())


# In[5]:


print(get_dict_len("dict.txt"))


# ### 划分训练集、验证集以及测试集

# In[ ]:


all_data_list = all_data.values.tolist()
train_length = len(all_data) // 10 * 7
dev_length = len(all_data) // 10 * 2

train_data = []
dev_data = []
test_data = []
for i in range(train_length):
    text = ""
    for s in all_data_list[i][1]:
        text = text + str(dict_txt[s]) + ","
    text = text[:-1]
    train_data.append([text, all_data_list[i][0]])

for i in range(train_length, train_length+dev_length):
    text = ""
    for s in all_data_list[i][1]:
        text = text + str(dict_txt[s]) + ","
    text = text[:-1]
    dev_data.append([text, all_data_list[i][0]])

for i in range(train_length+dev_length, len(all_data)):
    text = ""
    for s in all_data_list[i][1]:
        text = text + str(dict_txt[s]) + ","
    text = text[:-1]
    test_data.append([text, all_data_list[i][0]])

print(len(train_data))
print(len(dev_data))
print(len(test_data))
df_train = pd.DataFrame(columns=["text", "label"], data=train_data)
df_dev = pd.DataFrame(columns=["text", "label"], data=dev_data)
df_test = pd.DataFrame(columns=["text", "label"], data=test_data)
df_train.to_csv("train_data.csv", index=False)
df_dev.to_csv("dev_data.csv", index=False)
df_test.to_csv("test_data.csv", index=False)


# ### 自定义数据集

# In[6]:


import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import pandas as pd


# In[7]:


class MyDataset(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """
    def __init__(self, mode='train'):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        super(MyDataset, self).__init__()
        self.label = True
        if mode == 'train':
            text = pd.read_csv("train_data.csv")["text"].values.tolist()
            label = pd.read_csv("train_data.csv")["label"].values.tolist()
            self.data = []
            for i in range(len(text)):
                self.data.append([])
                self.data[-1].append(np.array([int(i) for i in text[i].split(",")]))
                self.data[-1][0] = self.data[-1][0][:256].astype('int64')if len(self.data[-1][0])>=256 else np.concatenate([self.data[-1][0], np.array([dict_txt["<unk>"]]*(256-len(self.data[-1][0])))]).astype('int64')
                self.data[-1].append(np.array(int(label[i])).astype('int64'))
        elif mode == 'dev':
            text = pd.read_csv("dev_data.csv")["text"].values.tolist()
            label = pd.read_csv("dev_data.csv")["label"].values.tolist()
            self.data = []
            for i in range(len(text)):
                self.data.append([])
                self.data[-1].append(np.array([int(i) for i in text[i].split(",")]))
                self.data[-1][0] = self.data[-1][0][:256].astype('int64')if len(self.data[-1][0])>=256 else np.concatenate([self.data[-1][0], np.array([dict_txt["<unk>"]]*(256-len(self.data[-1][0])))]).astype('int64')
                self.data[-1].append(np.array(int(label[i])).astype('int64'))
        else:
            text = pd.read_csv("test_data.csv")["text"].values.tolist()
            label = pd.read_csv("test_data.csv")["label"].values.tolist()
            self.data = []
            for i in range(len(text)):
                self.data.append([])
                self.data[-1].append(np.array([int(i) for i in text[i].split(",")]))
                self.data[-1][0] = self.data[-1][0][:256].astype('int64')if len(self.data[-1][0])>=256 else np.concatenate([self.data[-1][0], np.array([dict_txt["<unk>"]]*(256-len(self.data[-1][0])))]).astype('int64')
                self.data[-1].append(np.array(int(label[i])).astype('int64'))
            self.label = False
    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据，对应的标签）
        """
        text_ =  self.data[index][0]
        label_ = self.data[index][1]

        if self.label:
            return text_, label_
        else:
            return text_

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


# In[8]:


train_data = MyDataset(mode="train")
dev_data = MyDataset(mode="dev")
test_data = MyDataset(mode="test")


# In[9]:


BATCH_SIZE = 128


# In[10]:


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


#
# ### LSTM

# In[11]:


import paddle.nn as nn


# In[12]:


inputs_dim = get_dict_len("dict.txt")


# In[13]:


class myLSTM(nn.Layer):
    def __init__(self):
        super(myLSTM, self).__init__()

        # num_embeddings (int) - 嵌入字典的大小， input中的id必须满足 0 =< id < num_embeddings 。 。
        # embedding_dim (int) - 每个嵌入向量的维度。
        # padding_idx (int|long|None) - padding_idx的配置区间为 [-weight.shape[0], weight.shape[0]，如果配置了padding_idx，那么在训练过程中遇到此id时会被用
        # sparse (bool) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
        # weight_attr (ParamAttr|None) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 ParamAttr ，一般无需设置，默认值为None。
        self.embedding = nn.Embedding(inputs_dim, 256)

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
        self.lstm = nn.LSTM(256, 256, num_layers=2, direction='bidirectional',dropout=0.5)

        # in_features (int) – 线性变换层输入单元的数目。
        # out_features (int) – 线性变换层输出单元的数目。
        # weight_attr (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 ParamAttr 。
        # bias_attr (ParamAttr|bool, 可选) – 指定偏置参数的属性。 bias_attr 为bool类型且设置为False时，表示不会为该层添加偏置。 bias_attr 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 ParamAttr 。默认值为None。
        # name (str，可选) – 具体用法请参见 Name ，一般无需设置，默认值为None。
        self.linear = nn.Linear(in_features=256*2, out_features=2)

        self.dropout = nn.Dropout(0.5)


    def forward(self, inputs):

        emb = self.dropout(self.embedding(inputs))

        output, (hidden, _) = self.lstm(emb)
        #output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        #hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        #把前向的hidden与后向的hidden合并在一起
        hidden = paddle.concat((hidden[-2,:,:], hidden[-1,:,:]), axis = 1)
        hidden = self.dropout(hidden)
        #hidden形状大小为[batch_size, hidden_size * num_directions]
        return self.linear(hidden)



# #### 封装模型

# In[14]:


lstm_model = paddle.Model(myLSTM())


# #### 配置优化器等参数

# In[15]:


lstm_model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=lstm_model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())


# #### 模型训练

# In[16]:


lstm_model.fit(train_loader,
          dev_loader,
          epochs=10,
          batch_size=BATCH_SIZE,
          verbose=1,
          save_dir="work/lstm")


# #### 模型预测

# In[18]:


result = lstm_model.predict(test_loader)


# ### GRU

# In[29]:


class myGRU(nn.Layer):
    def __init__(self):
        super(myGRU, self).__init__()

        # num_embeddings (int) - 嵌入字典的大小， input中的id必须满足 0 =< id < num_embeddings 。 。
        # embedding_dim (int) - 每个嵌入向量的维度。
        # padding_idx (int|long|None) - padding_idx的配置区间为 [-weight.shape[0], weight.shape[0]，如果配置了padding_idx，那么在训练过程中遇到此id时会被用
        # sparse (bool) - 是否使用稀疏更新，在词嵌入权重较大的情况下，使用稀疏更新能够获得更快的训练速度及更小的内存/显存占用。
        # weight_attr (ParamAttr|None) - 指定嵌入向量的配置，包括初始化方法，具体用法请参见 ParamAttr ，一般无需设置，默认值为None。
        self.embedding = nn.Embedding(inputs_dim, 256)

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
        self.gru = nn.GRU(256, 256, num_layers=2, direction='bidirectional',dropout=0.5)

        # in_features (int) – 线性变换层输入单元的数目。
        # out_features (int) – 线性变换层输出单元的数目。
        # weight_attr (ParamAttr, 可选) – 指定权重参数的属性。默认值为None，表示使用默认的权重参数属性，将权重参数初始化为0。具体用法请参见 ParamAttr 。
        # bias_attr (ParamAttr|bool, 可选) – 指定偏置参数的属性。 bias_attr 为bool类型且设置为False时，表示不会为该层添加偏置。 bias_attr 如果设置为True或者None，则表示使用默认的偏置参数属性，将偏置参数初始化为0。具体用法请参见 ParamAttr 。默认值为None。
        # name (str，可选) – 具体用法请参见 Name ，一般无需设置，默认值为None。
        self.linear = nn.Linear(in_features=256*2, out_features=2)

        self.dropout = nn.Dropout(0.5)


    def forward(self, inputs):

        emb = self.dropout(self.embedding(inputs))

        output, hidden = self.gru(emb)
        #output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        #hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        #把前向的hidden与后向的hidden合并在一起
        hidden = paddle.concat((hidden[-2,:,:], hidden[-1,:,:]), axis = 1)
        hidden = self.dropout(hidden)
        #hidden形状大小为[batch_size, hidden_size * num_directions]
        return self.linear(hidden)



# #### 封装模型

# In[30]:


GRU_model = paddle.Model(myGRU())


# #### 配置优化器等参数

# In[31]:


GRU_model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=GRU_model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())


# #### 模型训练

# In[32]:


GRU_model.fit(train_loader,
          dev_loader,
          epochs=10,
          batch_size=BATCH_SIZE,
          verbose=1,
          save_dir="work/GRU")


# #### 模型预测

# In[33]:


result = GRU_model.predict(test_loader)