#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import argparse
import datetime
import collections

import numpy as np
import tensorflow as tf

"""
此例子中用到的数据是从 Tomas Mikolov 的网站取得的 PTB 数据集
PTB 文本数据集是语言模型学习中目前最广泛的数据集。
数据集中我们只需要利用 data 文件夹中的
ptb.test.txt，ptb.train.txt，ptb.valid.txt 三个数据文件
测试，训练，验证 数据集
这三个数据文件是已经经过预处理的，包含10000个不同的词语和语句结束标识符 <eos> 的
要获得此数据集，只需要用下面一行命令：
wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
如果没有 wget 的话，就安装一下：
sudo apt install wget
解压下载下来的压缩文件：
tar xvf simple-examples.tgz
==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭 代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代
# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch
==== 超参数（Hyper parameter）====
init_scale : 权重参数（Weights）的初始取值跨度，一开始取小一些比较利于训练
learning_rate : 学习率，训练时初始为 1.0
num_layers : LSTM 层的数目（默认是 2）
num_steps : LSTM 展开的步（step）数，相当于每个批次输入单词的数目（默认是 35）
hidden_size : LSTM 层的神经元数目，也是词向量的维度（默认是 650）
max_lr_epoch : 用初始学习率训练的 Epoch 数目（默认是 10）
dropout : 在 Dropout 层的留存率（默认是 0.5）
lr_decay : 在过了 max_lr_epoch 之后每一个 Epoch 的学习率的衰减率，训练时初始为 0.93。让学习率逐渐衰减是提高训练效率的有效方法
batch_size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目
（batch_size 默认是 20。取比较小的 batch_size 更有利于 Stochastic Gradient Descent（随机梯度下降），防止被困在局部最小值）
"""

# 数据集的目录
data_path = "data"

# 保存训练所得的模型参数文件的目录
save_path = './save'

# 测试时读取模型参数文件的名称
load_file = "train-checkpoint-69"

parser = argparse.ArgumentParser()
# 数据集的目录
parser.add_argument('--data_path', type=str, default=data_path, help='The path of the data for training and testing')
# 测试时读取模型参数文件的名称
parser.add_argument('--load_file', type=str, default=load_file, help='The path of checkpoint file of model variables saved during training')
args = parser.parse_args()

# 如果是 Python3 版本
Py3 = sys.version_info[0] == 3


# 将文件根据句末分割符 <eos> 来分割
def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()


# 构造从单词到唯一整数值的映射
# 后面的其他数的整数值按照它们在数据集里出现的次数多少来排序，出现较多的排前面
# 单词 the 出现频次最多，对应整数值是 0
# <unk> 表示 unknown（未知），第二多，整数值为 1
def build_vocab(filename):
    data = read_words(filename)

    # 用 Counter 统计单词出现的次数，为了之后按单词出现次数的多少来排序
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    # 单词到整数的映射
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


# 将文件里的单词都替换成独一的整数
def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


# 加载所有数据，读取所有单词，把其转成唯一对应的整数值
def load_data(data_path):
    # 确保包含所有数据集文件的 data_path 文件夹在所有 Python 文件
    # 的同级目录下。当然了，你也可以自定义文件夹名和路径
    if not os.path.exists(data_path):
        raise Exception("包含所有数据集文件的 {} 文件夹 不在此目录下，请添加".format(data_path))

    # 三个数据集的路径
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    # 建立词汇表，将所有单词（word）转为唯一对应的整数值（id）
    word_to_id = build_vocab(train_path)

    # 训练，验证和测试数据
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    # 所有不重复单词的个数
    vocab_size = len(word_to_id)

    # 反转一个词汇表：为了之后从 整数 转为 单词
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(word_to_id)
    print("===================")
    print(vocab_size)
    print("===================")
    print(train_data[:10])
    print("===================")
    print(" ".join([id_to_word[x] for x in train_data[:10]]))
    print("===================")
    return train_data, valid_data, test_data, vocab_size, id_to_word


# if __name__ == '__main__':
#     # 数据集的目录
#     data_path = "data"
#
#     # 保存训练所得的模型参数文件的目录
#     save_path = './save'
#
#     # 测试时读取模型参数文件的名称
#     load_file = "train-checkpoint-69"
#
#     parser = argparse.ArgumentParser()
#     # 数据集的目录
#     parser.add_argument('--data_path', type=str, default=data_path,
#                         help='The path of the data for training and testing')
#     # 测试时读取模型参数文件的名称
#     parser.add_argument('--load_file', type=str, default=load_file,
#                         help='The path of checkpoint file of model variables saved during training')
#     args = parser.parse_args()
#
#     # 如果是 Python3 版本
#     Py3 = sys.version_info[0] == 3
#
#     load_data(data_path)

# 生成批次样本
def generate_batches(raw_data, batch_size, num_steps):
    # 将数据转为 Tensor 类型
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size

    # 将数据形状转为 [batch_size, batch_len]
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    # range_input_producer 可以用多线程异步的方式从数据集里提取数据
    # 用多线程可以加快训练，因为 feed_dict 的赋值方式效率不高
    # shuffle 为 False 表示不打乱数据而按照队列先进先出的方式提取数据
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    # 假设一句话是这样： “我爱我的祖国和人民”
    # 那么，如果 x 是类似这样： “我爱我的祖国”
    x = data[:, i * num_steps:(i + 1) * num_steps]
    x.set_shape([batch_size, num_steps])
    # y 就是类似这样（正好是 x 的时间步长 + 1）： “爱我的祖国和”
    # 因为我们的模型就是要预测一句话中每一个单词的下一个单词
    # 当然这边的例子很简单，实际的数据不止一个维度
    y = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
    y.set_shape([batch_size, num_steps])

    return x, y


# 输入数据
class Input(object):
    def __init__(self, batch_size, num_steps, data):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        # input_data 是输入，targets 是期望的输出
        self.input_data, self.targets = generate_batches(data, batch_size, num_steps)


import tensorflow as tf


# 神经网络的模型
class Model(object):
    # 构造函数
    def __init__(self, input_obj, is_training, hidden_size, vocab_size, num_layers,
                 dropout=0.5, init_scale=0.05):
        self.is_training = is_training
        self.input_obj = input_obj
        self.batch_size = input_obj.batch_size
        self.num_steps = input_obj.num_steps
        self.hidden_size = hidden_size

        # 让这里的操作和变量用 CPU 来计算，因为暂时（貌似）还没有 GPU 的实现
        with tf.device("/cpu:0"):
            # 创建 词向量（Word Embedding），Embedding 表示 Dense Vector（密集向量）
            # 词向量本质上是一种单词聚类（Clustering）的方法
            embedding = tf.Variable(tf.random_uniform([vocab_size, self.hidden_size], -init_scale, init_scale))
            # embedding_lookup 返回词向量
            inputs = tf.nn.embedding_lookup(embedding, self.input_obj.input_data)

        # 如果是 训练时 并且 dropout 率小于 1，使输入经过一个 Dropout 层
        # Dropout 防止过拟合
        if is_training and dropout < 1:
            inputs = tf.nn.dropout(inputs, dropout)

        # 状态（state）的存储和提取
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
        # 一个是 前一时刻 LSTM 的输出 h(t-1)
        # 一个是 前一时刻的单元状态 C(t-1)
        # 这个 C 和 h 是用于构建之后的 tf.contrib.rnn.LSTMStateTuple
        self.init_state = tf.placeholder(tf.float32, [num_layers, 2, self.batch_size, self.hidden_size])

        # 每一层的状态
        state_per_layer_list = tf.unstack(self.init_state, axis=0)

        # 初始的状态（包含 前一时刻 LSTM 的输出 h(t-1) 和 前一时刻的单元状态 C(t-1)），用于之后的 dynamic_rnn
        rnn_tuple_state = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        # 创建一个 LSTM 层，其中的神经元数目是 hidden_size 个（默认 650 个）
        cell = tf.contrib.rnn.LSTMCell(hidden_size)

        # 如果是训练时 并且 Dropout 率小于 1，给 LSTM 层加上 Dropout 操作
        # 这里只给 输出 加了 Dropout 操作，留存率(output_keep_prob)是 0.5
        # 输入则是默认的 1，所以相当于输入没有做 Dropout 操作
        if is_training and dropout < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)

        # 如果 LSTM 的层数大于 1, 则总计创建 num_layers 个 LSTM 层
        # 并将所有的 LSTM 层包装进 MultiRNNCell 这样的序列化层级模型中
        # state_is_tuple=True 表示接受 LSTMStateTuple 形式的输入状态
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)], state_is_tuple=True)

        # dynamic_rnn（动态 RNN）可以让不同迭代传入的 Batch 可以是长度不同的数据
        # 但同一次迭代中一个 Batch 内部的所有数据长度仍然是固定的
        # dynamic_rnn 能更好处理 padding（补零）的情况，节约计算资源
        # 返回两个变量：
        # 第一个是一个 Batch 里在时间维度（默认是 35）上展开的所有 LSTM 单元的输出，形状默认为 [20, 35, 650]，之后会经过扁平层处理
        # 第二个是最终的 state（状态），包含 当前时刻 LSTM 的输出 h(t) 和 当前时刻的单元状态 C(t)
        output, self.state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, initial_state=rnn_tuple_state)

        # 扁平化处理，改变输出形状为 (batch_size * num_steps, hidden_size)，形状默认为 [700, 650]
        output = tf.reshape(output, [-1, hidden_size]) # -1 表示 自动推导维度大小

        # Softmax 的权重（Weight）
        softmax_w = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -init_scale, init_scale))
        # Softmax 的偏置（Bias）
        softmax_b = tf.Variable(tf.random_uniform([vocab_size], -init_scale, init_scale))

        # logits 是 Logistic Regression（用于分类）模型（线性方程： y = W * x + b ）计算的结果（分值）
        # 这个 logits（分值）之后会用 Softmax 来转成百分比概率
        # output 是输入（x）， softmax_w 是 权重（W），softmax_b 是偏置（b）
        # 返回 W * x + b 结果
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # 将 logits 转化为三维的 Tensor，为了 sequence loss 的计算
        # 形状默认为 [20, 35, 10000]
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

        # 计算 logits 的序列的交叉熵（Cross-Entropy）的损失（loss）
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,  # 形状默认为 [20, 35, 10000]
            self.input_obj.targets,  # 期望输出，形状默认为 [20, 35]
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)

        # 更新代价（cost）
        self.cost = tf.reduce_sum(loss)

        # Softmax 算出来的概率
        self.softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))

        # 取最大概率的那个值作为预测
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)

        # 预测值和真实值（目标）对比
        correct_prediction = tf.equal(self.predict, tf.reshape(self.input_obj.targets, [-1]))

        # 计算预测的精度
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 如果是 测试，则直接退出
        if not is_training:
            return

        # 学习率。trainable=False 表示“不可被训练”
        self.learning_rate = tf.Variable(0.0, trainable=False)

        # 返回所有可被训练（trainable=True。如果不设定 trainable=False，默认的 Variable 都是可以被训练的）
        # 也就是除了不可被训练的 学习率 之外的其他变量
        tvars = tf.trainable_variables()

        # tf.clip_by_global_norm（实现 Gradient Clipping（梯度裁剪））是为了防止梯度爆炸
        # tf.gradients 计算 self.cost 对于 tvars 的梯度（求导），返回一个梯度的列表
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        # 优化器用 GradientDescentOptimizer（梯度下降优化器）
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # apply_gradients（应用梯度）将之前用（Gradient Clipping）梯度裁剪过的梯度 应用到可被训练的变量上去，做梯度下降
        # apply_gradients 其实是 minimize 方法里面的第二步，第一步是 计算梯度
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        # 用于更新 学习率
        self.new_lr = tf.placeholder(tf.float32, shape=[])
        self.lr_update = tf.assign(self.learning_rate, self.new_lr)

    # 更新 学习率
    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


from utils import *
from network import *


def train(train_data, vocab_size, num_layers, num_epochs, batch_size, model_save_name,
          learning_rate=1.0, max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    # 训练的输入
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)

    # 创建训练的模型
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocab_size, num_layers=num_layers)

    # 初始化变量的操作
    init_op = tf.global_variables_initializer()

    # 初始的学习率（learning rate）的衰减率
    orig_decay = lr_decay

    with tf.Session() as sess:
        sess.run(init_op)  # 初始化所有变量

        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 为了用 Saver 来保存模型的变量
        saver = tf.train.Saver() # max_to_keep 默认是 5, 只保存最近的 5 个模型参数文件

        # 开始 Epoch 的训练
        for epoch in range(num_epochs):
            # 只有 Epoch 数大于 max_lr_epoch（设置为 10）后，才会使学习率衰减
            # 也就是说前 10 个 Epoch 的学习率一直是 1, 之后每个 Epoch 学习率都会衰减
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0)
            m.assign_lr(sess, learning_rate * new_lr_decay)

            # 当前的状态
            # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
            # 一个是 前一时刻 LSTM 的输出 h(t-1)
            # 一个是 前一时刻的单元状态 C(t-1)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))

            # 获取当前时间，以便打印日志时用
            curr_time = datetime.datetime.now()

            for step in range(training_input.epoch_size):
                # train_op 操作：计算被修剪（clipping）过的梯度，并最小化 cost（误差）
                # state 操作：返回时间维度上展开的最后 LSTM 单元的输出（C(t) 和 h(t)），作为下一个 Batch 的输入状态
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state], feed_dict={m.init_state: current_state})
                else:
                    seconds = (float((datetime.datetime.now() - curr_time).seconds) / print_iter)
                    curr_time = datetime.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                    # 每 print_iter（默认是 50）打印当下的 Cost（误差/损失）和 Accuracy（精度）
                    print("Epoch {}, 第 {} 步, 损失: {:.3f}, 精度: {:.3f}, 每步所用秒数: {:.2f}".format(epoch, step, cost, acc, seconds))

            # 保存一个模型的变量的 checkpoint 文件
            saver.save(sess, save_path + '/' + model_save_name, global_step=epoch)
        # 对模型做一次总的保存
        saver.save(sess, save_path + '/' + model_save_name + '-final')

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    train(train_data, vocab_size, num_layers=2, num_epochs=70, batch_size=20,
          model_save_name='train-checkpoint')


from utils import *
from network import *


def test(model_path, test_data, vocab_size, id_to_word):
    # 测试的输入
    test_input = Input(batch_size=20, num_steps=35, data=test_data)

    # 创建测试的模型，基本的超参数需要和训练时用的一致，例如：
    # hidden_size，num_steps，num_layers，vocab_size，batch_size 等等
    # 因为我们要载入训练时保存的参数的文件，如果超参数不匹配 TensorFlow 会报错
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, num_layers=2)

    # 为了用 Saver 来恢复训练时生成的模型的变量
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 当前的状态
        # 第二维是 2 是因为测试时指定只有 2 层 LSTM
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
        # 一个是 前一时刻 LSTM 的输出 h(t-1)
        # 一个是 前一时刻的单元状态 C(t-1)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        # 恢复被训练的模型的变量
        saver.restore(sess, model_path)

        # 测试 30 个批次
        num_acc_batches = 30

        # 打印预测单词和实际单词的批次数
        check_batch_idx = 25

        # 超过 5 个批次才开始累加精度
        acc_check_thresh = 5

        # 初始精度的和，用于之后算平均精度
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                pred_words = [id_to_word[x] for x in pred[:m.num_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print("\n实际的单词:")
                print(" ".join(true_words))  # 真实的单词
                print("预测的单词:")
                print(" ".join(pred_words))  # 预测的单词
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc

        # 打印平均精度
        print("平均精度: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    if args.load_file:
        load_file = args.load_file
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    trained_model = save_path + "/" + load_file

    test(trained_model, test_data, vocab_size, id_to_word)

