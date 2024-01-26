#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('unzip -o data/data36219/text8.zip')


# In[2]:


from paddle import fluid
import numpy as np
import time


class LSTM:
    """
    emb_dim: 词向量维度
    vocab_size: 词典大小，不能小于训练数据中所有词的总数
    num_layers: 隐含层的数量
    hidden_size: 隐含层的大小
    num_steps: LSTM 一次接收数据的最大长度，样本的timestamp
    use_gpu: 是否使用gpu进行训练
    dropout_prob: 如果大于0，就启用dropout，值在0-1区间
    init_scale: 训练参数的初始化范围
    lr：学习速率
    vocab: 默认为None，占位，暂时没用
    """

    def __init__(self,
                 vocab_size,
                 num_layers,
                 hidden_size,
                 num_steps,
                 use_gpu=True,
                 dropout_prob=None,
                 init_scale=0.1,
                 lr=0.001,
                 vocab=None):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.dropout_prob = dropout_prob
        self.use_gpu = use_gpu
        self.init_scale = init_scale
        self.vocab = vocab
        self.lr = lr

    def forward(self, x, batch_size):
        self.init_hidden = fluid.layers.data(name='init_hidden',
                                             shape=[self.num_layers, batch_size, self.hidden_size],
                                             append_batch_size=False)
        self.init_cell = fluid.layers.data(name='init_cell',
                                           shape=[self.num_layers, batch_size, self.hidden_size],
                                           append_batch_size=False)
        x_emb = fluid.embedding(input=x, size=[self.vocab_size, self.hidden_size],
                                dtype='float32', is_sparse=False,
                                param_attr=fluid.ParamAttr(
                                    name='embedding_para',
                                    initializer=fluid.initializer.UniformInitializer(
                                        low=-self.init_scale, high=self.init_scale
                                    )
                                ))
        x_emb = fluid.layers.reshape(x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout_prob is not None and self.dropout_prob > 0.0:
            x_emb = fluid.layers.dropout(x_emb, dropout_prob=self.dropout_prob,
                                         dropout_implementation="upscale_in_train")

        rnn_out, last_hidden, last_cell = fluid.contrib.layers.basic_lstm(x_emb, self.init_hidden, self.init_cell,
                                                                          self.hidden_size, self.num_layers,
                                                                          dropout_prob=self.dropout_prob)
        rnn_out = fluid.layers.reshape(rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        softmax_weight = fluid.layers.create_parameter(
            [self.hidden_size, self.vocab_size],
            dtype="float32",
            name="softmax_weight",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        softmax_bias = fluid.layers.create_parameter(
            [self.vocab_size],
            dtype="float32",
            name='softmax_bias',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

        proj = fluid.layers.matmul(rnn_out, softmax_weight)
        proj = fluid.layers.elementwise_add(proj, softmax_bias)
        proj = fluid.layers.reshape(proj, shape=[-1, self.vocab_size], inplace=True)
        # 更新 init_hidden, init_cell
        fluid.layers.assign(input=last_cell, output=self.init_cell)
        fluid.layers.assign(input=last_hidden, output=self.init_hidden)
        return proj, last_hidden, last_cell

    def train(self, x, epochs=3, batch_size=32, log_interval=100):
        """
        :param log_interval: 输出信息的间隔
        :param x: 输入，一维list，文本需要经过编码
        :param epochs: 训练回合数
        :param batch_size: 训练batch大小
        :return:
        """
        self.batch_size = batch_size
        # 定义训练的program
        main_program = fluid.default_main_program()
        startup_program = fluid.default_startup_program()
        train_loss, train_proj, self.last_hidden, self.last_cell, py_reader = self.build_train_model(main_program, startup_program)

        # 定义测试的program, 写成全局的，以便留给测试函数
        self.test_program = fluid.Program()
        self.test_startup_program = fluid.Program()
        self.test_loss, self.test_proj, _, _ = self.build_test_model(self.test_program, self.test_startup_program)
        self.test_program = self.test_program.clone(for_test=True)

        place = fluid.CUDAPlace(0) if self.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(startup_program)

        def data_gen():
            batches = self.get_data_iter(x)
            for batch in batches:
                x_, y_ = batch
                yield x_, y_

        py_reader.decorate_tensor_provider(data_gen)

        for epoch in range(epochs):
            batch_times = []
            epoch_start_time = time.time()
            total_loss = 0
            iters = 0
            py_reader.start()
            batch_id = 0
            batch_start_time = time.time()
            # 初始化init_hidden, init_cell
            init_hidden = np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32')
            init_cell = np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32')

            data_len = len(x)
            batch_len = data_len // self.batch_size
            batch_num = (batch_len - 1) // self.num_steps

            # 送入数据，抓取结果
            try:
                while True:
                    # 送入数据
                    data_feeds = {}
                    data_feeds['init_hidden'] = init_hidden
                    data_feeds['init_cell'] = init_cell
                    fetch_outs = self.exe.run(main_program, feed=data_feeds,
                                         fetch_list=[train_loss.name, self.last_hidden.name, self.last_cell.name])
                    t_loss = np.array(fetch_outs[0])
                    init_hidden = np.array(fetch_outs[1])
                    init_cell = np.array(fetch_outs[2])

                    total_loss += t_loss
                    batch_time = time.time() - batch_start_time
                    batch_times.append(batch_time)
                    batch_start_time = time.time()

                    batch_id += 1
                    iters += self.num_steps
                    if batch_id % log_interval == 0:
                        ppl = np.exp(total_loss / iters)
                        print("-- Epoch: %d - Batch: %d / %d - Cost Time: %.2f s -ETA: %.2f s- ppl: %.5f"
                              % (epoch + 1, batch_id, batch_num, sum(batch_times),
                                 sum(batch_times) / batch_id * (batch_num - batch_id), ppl[0]))
            except fluid.core.EOFException:
                py_reader.reset()

            epoch_time = time.time() - epoch_start_time
            ppl = np.exp(total_loss / iters)
            print("Epoch %d Done. Cost Time: %.2f s. ppl: %.5f." % (epoch + 1, epoch_time, ppl))

    def evaluate(self, x):
        """
        测试模型的效果
        :param x:
        :return:
        """
        eval_data_gen = self.get_data_iter(x)
        total_loss = 0.0
        iters = 0
        # 初始化init_hidden, init_cell
        init_hidden = np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32')
        init_cell = np.zeros((self.num_layers, self.batch_size, self.hidden_size), dtype='float32')

        for batch_id, batch in enumerate(eval_data_gen):
            x, y = batch
            data_feeds = {}
            data_feeds['init_hidden'] = init_hidden
            data_feeds['init_cell'] = init_cell
            data_feeds['x'] = x
            data_feeds['y'] = y
            fetch_outs = self.exe.run(self.test_program, feed=data_feeds,
                                      fetch_list=[self.test_loss.name, self.last_hidden.name, self.last_cell.name])
            cost_test = np.array(fetch_outs[0])
            init_hidden = np.array(fetch_outs[1])
            init_cell = np.array(fetch_outs[2])

            total_loss += cost_test
            iters += self.num_steps
            ppl = np.exp(total_loss / iters)
            print("-- Batch: %d - ppl: %.5f" % (batch_id, ppl[0]))
        print("ppl: %.5f" % (ppl[0]))
        return ppl

    def get_data_iter(self, raw_data):
        """
        处理原始文本，生成训练数据
        对于RNN来说，一般为读取前n个词，然后预测下一个词，这里简化为每读一个词，预测下一个词。
        由于LSTM考虑了长依赖，所以也可以做到读取n个词，预测下一个词
        :param raw_data: 一个一维数组，list，
        :return:
        """
        data_len = len(raw_data)
        raw_data = np.asarray(raw_data, dtype='int64')
        batch_len = data_len // self.batch_size
        # 将一维数组变为二维数组，第一维是batch的数量，第二维是每个batch的数据，这里对后边不足batch_len的数据进行了裁剪，弃掉不用
        data = raw_data[0:self.batch_size * batch_len].reshape((self.batch_size, batch_len))

        # 为了保证每个batch最后一个词能够被预测，x的词最多被分到batch_len-1
        batch_num = (batch_len - 1) // self.num_steps
        for i in range(batch_num):
            x = np.copy(data[:, i * self.num_steps:(i + 1) * self.num_steps])
            y = np.copy(data[:, i * self.num_steps + 1:(i + 1) * self.num_steps + 1])
            x = x.reshape((-1, self.num_steps, 1))
            y = y.reshape((-1, 1))
            yield x, y

    def build_train_model(self, main_program, startup_program):
        """
        读取数据，构建网络
        :param main_program:
        :param startup_program:
        :return:
        """
        with fluid.program_guard(main_program, startup_program):
            feed_shapes = [[self.batch_size, self.num_steps, 1],
                           [self.batch_size * self.num_steps, 1]]
            py_reader = fluid.layers.py_reader(capacity=64, shapes=feed_shapes, dtypes=['int64', 'int64'])
            x, y = fluid.layers.read_file(py_reader)
            # 使用unique_name.guard创建变量空间，以便在test时共享参数
            with fluid.unique_name.guard():
                proj, last_hidden, last_cell = self.forward(x, self.batch_size)

                loss = self.get_loss(proj, y)
                optimizer = fluid.optimizer.Adam(learning_rate=self.lr,
                                                 grad_clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=1000))
                optimizer.minimize(loss)

                # 不知道有什么用，先写上
                #loss.persistable = True
                #proj.persistable = True
                #last_cell.persistable = True
                #last_hidden.persistable = True

                return loss, proj, last_hidden, last_cell, py_reader

    def build_test_model(self, main_program, startup_program):
        """
        验证模型效果
        :param main_program:
        :param startup_program:
        :return:
        """
        with fluid.program_guard(main_program, startup_program):
            x = fluid.layers.data(name='x', shape=[self.batch_size, self.num_steps, 1], dtype='int64', append_batch_size=False)
            y = fluid.layers.data(name='y', shape=[self.batch_size * self.num_steps, 1], dtype='int64', append_batch_size=False)
            # 使用unique_name.guard创建变量空间，和train共享参数
            with fluid.unique_name.guard():
                proj, last_hidden, last_cell = self.forward(x, self.batch_size)
                loss = self.get_loss(proj, y)

                # 不知道有什么用，先写上
                #loss.persistable = True
                #proj.persistable = True
                #last_cell.persistable = True
                #last_hidden.persistable = True

                return loss, proj, last_hidden, last_cell

    def get_loss(self, proj, y):
        loss = fluid.layers.softmax_with_cross_entropy(logits=proj, label=y, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)
        return loss


# ### 1.3.1 数据预处理

# In[3]:


import re
from collections import Counter
import itertools

def clean_str(string):
    """
    将文本中的特定字符串做修改和替换处理
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
    string = re.sub(r":", " : ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(sentences, EOS='</eos>'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}
    vocabulary[EOS] = 0
    return [vocabulary, vocabulary_inv]


def file_to_ids(src_file, src_vocab):
    """
    将文章单词序列转化成词典id序列
    :param src_file:
    :param src_vocab:
    :return:
    """
    src_data = []
    for line in src_file:
        ids = [src_vocab[w] for w in line if w in src_vocab]
        src_data += ids + [0]
    return src_data


# In[4]:


x_text = list(open("text8", "r").readlines())
x_text = [clean_str(sent) for sent in x_text]
vocabulary, vocabulary_inv = build_vocab(x_text)
x_text = file_to_ids(x_text, vocabulary)


# ### 1.3.2 训练 
# 训练的结果与数据的吻合程度用困惑度指标ppl来衡量，参考了[基于LSTM的语言模型实现](https://aistudio.baidu.com/aistudio/projectdetail/592038)。ppl的值即e为底，平均交叉熵损失为指数的幂指数值。

# In[5]:


lstm_test = LSTM(vocab_size=len(vocabulary), num_layers=1, hidden_size=100, num_steps=20, use_gpu=True, dropout_prob=0.2, init_scale=0.1, lr=0.01)
lstm_test.train(x_text[:1000000], epochs=3, batch_size=32, log_interval=100)


# ### 1.3.3 测试结果

# In[6]:


ppl = lstm_test.evaluate(x_text[1000000:1005000])


# In[8]:


ppl = gru_test.evaluate(x_text[1000000:1005000])

