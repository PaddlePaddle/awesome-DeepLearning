#!/usr/bin/env python
# coding: utf-8

# # Text matching with LSTM model on LCQMC dataset.
# # 1. 实验设计逻辑
# 文本语义匹配任务，简单来说就是给定两段文本，让模型来判断两段文本是不是语义相似,以下两个句子被认为是相互匹配的：
# 
# > 句子1：我不爱吃烤冷面，但是我爱吃冷面
# >
# > 句子2：我爱吃冷面，但是不爱吃烤冷面
# Text matching with LSTM model on LCQMC dataset.
# 
# 本案例在权威语义匹配数据集 LCQMC 上基于LSTM实现文本匹配。
# ![](https://ai-studio-static-online.cdn.bcebos.com/34bf5e8eb10341ea8897ac3bcd79166d0d39a7d9844342fda9656a60bd3c29c6)
# 
# 实验设计基本逻辑是使用LSTM网络，把每个句子抽象成一个向量表示，通过计算这两个向量之间的相似度，就可以快速完成文本相似度计算任务。在实际场景里，我们也通常使用LSTM网络的最后一步hidden结果，将一个句子抽象成一个向量，然后通过向量点积，或者cosine相似度的方式，去衡量两个句子的相似度。

# # 2. 数据处理
# 本案例在权威语义匹配数据集 LCQMC 上基于LSTM实现文本匹配。 LCQMC 数据集是基于百度知道相似问题推荐构造的通问句语义匹配数据集。训练集中的每两段文本都会被标记为 1（语义相似） 或者 0（语义不相似）。
# ## 2.1 数据加载
# 此案例我们使用 PaddleNLP 内置的语义数据集 LCQMC 来进行训练、评估、预测。
# 
# 训练集 train.tsv: 用来训练模型参数的数据集，模型直接根据训练集来调整自身参数以获得更好的分类效果。
# 
# 验证集dev.tsv: 用于在训练过程中检验模型的状态，收敛情况。验证集通常用于调整超参数，根据几组模型验证集上的表现，决定采用哪组超参数。
# 
# 测试集test.tsv: 用来计算模型的各项评估指标，验证模型泛化能力。
# 
# LCQMC 数据集是公开的语义匹配权威数据集。PaddleNLP 已经内置该数据集，一键加载。

# In[1]:


get_ipython().system('pip install -U paddlepaddle -i https://mirror.baidu.com/pypi/simple')


# In[2]:


# 安装最新版本的 paddlenlp
get_ipython().system('pip install --upgrade paddlenlp -i https://pypi.org/simple')


# In[3]:


import time
import os
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp

# 加载 Lcqmc 的训练集、验证集
train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])


# In[4]:


# 输出训练集的前 10 条样本
for id, example in enumerate(train_ds):
    if id <= 10:
        print(example)


# ## 2.2 数据预处理
# 
# 通过 PaddleNLP 加载进来的 LCQMC 数据集是原始的明文数据集，这部分我们来实现组 batch、tokenize 等预处理逻辑，将原始明文数据转换成网络训练的输入数据。
# 
# - 定义样本转换函数

# In[5]:


tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')


# - 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据

# In[6]:


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids


# - 为了后续方便使用，我们使用python偏函数（partial）给 convert_example 赋予一些默认参数

# In[7]:


from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512)


# - 组装 Batch 数据 & Padding

# In[8]:


from paddlenlp.data import Stack, Pad, Tuple
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
print("Stacked Data: \n", result)

a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
print("Padded Data: \n", result)

data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn = Tuple(Pad(pad_val=0), Stack())
ids, labels = batchify_fn(data)


# In[9]:


# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]


# - 定义 Dataloader：基于组 batchify_fn 函数和样本转换函数 trans_func 来构造训练集的 DataLoader, 支持多卡训练

# In[10]:


# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


# # 3. 模型设计
# ## 3.1 定义长短时记忆模型
# ![](https://ai-studio-static-online.cdn.bcebos.com/8f227256e64c45e681629a855ad3dafcb9fa9d3e3c8e46098353892295834300)
# 

# In[11]:


import paddle.fluid as fluid
#使用飞桨实现一个长短时记忆模型
class SimpleLSTMRNN(fluid.Layer):
    
    def __init__(self,
                 hidden_size,
                 num_steps,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None):
        
        #这个模型有几个参数：
        #1. hidden_size，表示embedding-size，或者是记忆向量的维度
        #2. num_steps，表示这个长短时记忆网络，最多可以考虑多长的时间序列
        #3. num_layers，表示这个长短时记忆网络内部有多少层，我们知道，
        # 给定一个形状为[batch_size, seq_len, embedding_size]的输入，
        # 长短时记忆网络会输出一个同样为[batch_size, seq_len, embedding_size]的输出，
        # 我们可以把这个输出再链到一个新的长短时记忆网络上
        # 如此叠加多层长短时记忆网络，有助于学习更复杂的句子甚至是篇章。
        #4. init_scale，表示网络内部的参数的初始化范围，
        # 长短时记忆网络内部用了很多tanh，sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果，
        
        super(SimpleLSTMRNN, self).__init__()
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._init_scale = init_scale
        self._dropout = dropout
        self._input = None
        self._num_steps = num_steps
        self.cell_array = []
        self.hidden_array = []

        # weight_1_arr用于存储不同层的长短时记忆网络中，不同门的W参数
        self.weight_1_arr = []
        self.weight_2_arr = []
        # bias_arr用于存储不同层的长短时记忆网络中，不同门的b参数
        self.bias_arr = []
        self.mask_array = []

        # 通过使用create_parameter函数，创建不同长短时记忆网络层中的参数
        # 通过上面的公式，我们知道，我们总共需要8个形状为[_hidden_size, _hidden_size]的W向量
        # 和4个形状为[_hidden_size]的b向量，因此，我们在声明参数的时候，
        # 一次性声明一个大小为[self._hidden_size * 2, self._hidden_size * 4]的参数
        # 和一个 大小为[self._hidden_size * 4]的参数，这样做的好处是，
        # 可以使用一次矩阵计算，同时计算8个不同的矩阵乘法
        # 以便加快计算速度
        for i in range(self._num_layers):
            weight_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 2, self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-self._init_scale, high=self._init_scale))
            self.weight_1_arr.append(self.add_parameter('w_%d' % i, weight_1))
            bias_1 = self.create_parameter(
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-self._init_scale, high=self._init_scale)),
                shape=[self._hidden_size * 4],
                dtype="float32",
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_arr.append(self.add_parameter('b_%d' % i, bias_1))

    # 定义LSTM网络的前向计算逻辑，飞桨会自动根据前向计算结果，给出反向结果
    def forward(self, input_embedding, init_hidden=None, init_cell=None):
        self.cell_array = []
        self.hidden_array = []
        
        #输入有三个信号：
        # 1. input_embedding，这个就是输入句子的embedding表示，
        # 是一个形状为[batch_size, seq_len, embedding_size]的张量
        # 2. init_hidden，这个表示LSTM中每一层的初始h的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空
        # 3. init_cell，这个表示LSTM中每一层的初始c的值，有时候，
        # 我们需要显示地指定这个值，在不需要的时候，就可以把这个值设置为空

        # 我们需要通过slice操作，把每一层的初始hidden和cell值拿出来，
        # 并存储在cell_array和hidden_array中
        for i in range(self._num_layers):
            pre_hidden = fluid.layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = fluid.layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = fluid.layers.reshape(
                pre_hidden, shape=[-1, self._hidden_size])
            pre_cell = fluid.layers.reshape(
                pre_cell, shape=[-1, self._hidden_size])
            self.hidden_array.append(pre_hidden)
            self.cell_array.append(pre_cell)

        # res记录了LSTM中每一层的输出结果（hidden）
        res = []
        for index in range(self._num_steps):
            # 首先需要通过slice函数，拿到输入tensor input_embedding中当前位置的词的向量表示
            # 并把这个词的向量表示转换为一个大小为 [batch_size, embedding_size]的张量
            self._input = fluid.layers.slice(
                input_embedding, axes=[1], starts=[index], ends=[index + 1])
            self._input = fluid.layers.reshape(
                self._input, shape=[-1, self._hidden_size])
            
            # 计算每一层的结果，从下而上
            for k in range(self._num_layers):
                # 首先获取每一层LSTM对应上一个时间步的hidden，cell，以及当前层的W和b参数
                pre_hidden = self.hidden_array[k]
                pre_cell = self.cell_array[k]
                weight_1 = self.weight_1_arr[k]
                bias = self.bias_arr[k]

                # 我们把hidden和拿到的当前步的input拼接在一起，便于后续计算
                nn = fluid.layers.concat([self._input, pre_hidden], 1)
                
                # 将输入门，遗忘门，输出门等对应的W参数，和输入input和pre-hidden相乘
                # 我们通过一步计算，就同时完成了8个不同的矩阵运算，提高了运算效率
                gate_input = fluid.layers.matmul(x=nn, y=weight_1)

                # 将b参数也加入到前面的运算结果中
                gate_input = fluid.layers.elementwise_add(gate_input, bias)
                
                # 通过split函数，将每个门得到的结果拿出来
                i, j, f, o = fluid.layers.split(
                    gate_input, num_or_sections=4, dim=-1)
                
                # 把输入门，遗忘门，输出门等对应的权重作用在当前输入input和pre-hidden上
                c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(
                    i) * fluid.layers.tanh(j)
                m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)
                
                # 记录当前步骤的计算结果，
                # m是当前步骤需要输出的hidden
                # c是当前步骤需要输出的cell
                self.hidden_array[k] = m
                self.cell_array[k] = c
                self._input = m
                
                # 一般来说，我们有时候会在LSTM的结果结果内加入dropout操作
                # 这样会提高模型的训练鲁棒性
                if self._dropout is not None and self._dropout > 0.0:
                    self._input = fluid.layers.dropout(
                        self._input,
                        dropout_prob=self._dropout,
                        dropout_implementation='upscale_in_train')
                    
            res.append(
                fluid.layers.reshape(
                    self._input, shape=[1, -1, self._hidden_size]))
        
        # 计算长短时记忆网络的结果返回回来，包括：
        # 1. real_res：每个时间步上不同层的hidden结果
        # 2. last_hidden：最后一个时间步中，每一层的hidden的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        # 3. last_cell：最后一个时间步中，每一层的cell的结果，
        # 形状为：[batch_size, num_layers, hidden_size]
        real_res = fluid.layers.concat(res, 0)
        real_res = fluid.layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = fluid.layers.concat(self.hidden_array, 1)
        last_hidden = fluid.layers.reshape(
            last_hidden, shape=[-1, self._num_layers, self._hidden_size])
        last_hidden = fluid.layers.transpose(x=last_hidden, perm=[1, 0, 2])
        last_cell = fluid.layers.concat(self.cell_array, 1)
        last_cell = fluid.layers.reshape(
            last_cell, shape=[-1, self._num_layers, self._hidden_size])
        last_cell = fluid.layers.transpose(x=last_cell, perm=[1, 0, 2])
        
        return real_res, last_hidden, last_cell


# ## 3.2 定义基于LSTM的TextMatching模型
# 
# 使用LSTM网络，把每个句子抽象成一个向量表示，通过计算这两个向量之间的相似度，就可以快速完成文本相似度计算任务。在实际场景里，我们也通常使用LSTM网络的最后一步hidden结果，将一个句子抽象成一个向量，然后通过向量点积，或者cosine相似度的方式，去衡量两个句子的相似度。
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/96085451c1a448128632f51dcb384f86677e8a7cd0b04af9a319b6e3efb837f0)
# 

# In[12]:


import paddle.nn as nn

# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
#pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')


class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)

        return probs

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)


# # 4. 训练配置
# 
# - 定义模型训练的超参数,模型实例化,指定训练的 cpu 或 gpu 资源,定义优化器等等

# In[14]:


from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 3
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.0,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()


# # 5. 模型训练与评估
# 
# - 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数。

# In[15]:



@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        losses.append(loss.numpy())
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()


# - 开始正式训练模型

# In[ ]:


global_step = 0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):

        input_ids, token_type_ids, labels = batch
        probs = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(probs, labels)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        
        # 每间隔 10 step 输出训练指标
        if global_step % 10 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        # 每间隔 100 step 在验证集和测试集上进行评估
        if global_step % 100 == 0:
            evaluate(model, criterion, metric, dev_data_loader, "dev")
            
# 训练结束后，存储模型参数
save_dir = os.path.join("checkpoint", "model_%d" % global_step)
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state.pdparams')
paddle.save(model.state_dict(), save_param_path)
tokenizer.save_pretrained(save_dir)


# # 6. 模型推理
# 设计一个接口函数,通过这个接口函数能够方便地对任意一个样本进行实时预测。使用已经训练好的语义匹配模型对一些预测数据进行预测。待预测数据为每行都是文本对的 tsv 文件，我们使用 Lcqmc 数据集的测试集作为我们的预测数据。

# In[ ]:


# 下载我们基于 Lcqmc 事先训练好的语义匹配模型并解压
get_ipython().system(' wget https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar')
get_ipython().system(' tar -xvf ernie_gram_zh_pointwise_matching_model.tar')


# - 定义预测函数

# In[ ]:


def predict(model, data_loader):
    
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs


# - 定义预测数据的 data_loader

# In[ ]:


# 预测数据的转换函数
# predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512,
    is_test=True)

# 预测数据的组 batch 操作
# predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
): [data for data in fn(samples)]

# 加载预测数据
test_ds = load_dataset("lcqmc", splits=["test"])


# In[ ]:


batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)

# 生成预测数据 data_loader
predict_data_loader =paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


# - 定义预测模型

# In[ ]:


pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')

model = PointwiseMatching(pretrained_model)


# - 加载已训练好的模型参数

# In[ ]:


# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
state_dict = paddle.load("./ernie_gram_zh_pointwise_matching_model/model_state.pdparams")

# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
# state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)


# - 开始预测

# In[ ]:


for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)


# In[ ]:


# 执行预测函数
y_probs = predict(model, predict_data_loader)

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)


# - 输出预测结果

# In[9]:


test_ds = load_dataset("lcqmc", splits=["test"])

with open("lcqmc.tsv", 'w', encoding="utf-8") as f:
    f.write("index\tprediction\n")    
    for idx, y_pred in enumerate(y_preds):
        f.write("{}\t{}\n".format(idx, y_pred))
        text_pair = test_ds[idx]
        text_pair["label"] = y_pred
        print(text_pair)

