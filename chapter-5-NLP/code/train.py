# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import paddle
from model import word2vec
from utils import utils
from utils import data_processor


# 处理数据，获得方便传入模型的数据形式
def get_dataset(data_path, corpus_rate=1.0, subsampling=True):
    # 加载数据
    corpus = data_processor.load_data(data_path)
    # 对语料进行预处理
    corpus = data_processor.data_preprocess(corpus)
    corpus = corpus[:int(len(corpus) * corpus_rate)]
    # 根据语料构造字典，统计每个词的频率，并根据频率将每个词转换为一个整数id
    word2id_freq, word2id_dict, id2word_dict = data_processor.build_dict(corpus)
    # 将语料转换为ID序列
    corpus = data_processor.convert_corpus_to_id(corpus, word2id_dict)
    # 使用二次采样算法处理语料，强化训练效果
    if subsampling:
        corpus = data_processor.subsampling(corpus, word2id_freq)
    # 构造数据，准备模型训练
    dataset = data_processor.build_data(corpus, word2id_dict)

    return dataset, word2id_dict, id2word_dict

def train(model, data_loader):
    # 开始训练，定义一些训练过程中需要使用的超参数
    batch_size = 128
    epoch_num = 3
    embedding_size = 200
    step = 0
    learning_rate = 0.001

    # 判断可用的模型训练环境，优先使用GPU
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    # 开启模型训练模式
    model.train()

    # 构造训练这个网络的优化器
    adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

    # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
    for center_words, target_words, label in data_loader:
        # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
        center_words_var = paddle.to_tensor(center_words)
        target_words_var = paddle.to_tensor(target_words)
        label_var = paddle.to_tensor(label)

        # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
        pred, loss = model(center_words_var, target_words_var, label_var)

        # 程序自动完成反向计算
        loss.backward()
        # 程序根据loss，完成一步对参数的优化更新
        adam.step()
        # 清空模型中的梯度，以便于下一个mini-batch进行更新
        adam.clear_grad()

        # 每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
        step += 1
        if step % 1000 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

        # 每隔10000步，打印一次模型对以下查询词的相似词，这里我们使用词和词之间的向量点积作为衡量相似度的方法，只打印了5个最相似的词
        if step % 10000 == 0:
            utils.get_similar_tokens('movie', 5, model.embedding.weight, word2id_dict, id2word_dict)
            utils.get_similar_tokens('one', 5, model.embedding.weight, word2id_dict, id2word_dict)
            utils.get_similar_tokens('chip', 5, model.embedding.weight, word2id_dict, id2word_dict)


if __name__ == '__main__':
    # 开始训练，定义一些训练过程中需要使用的超参数
    batch_size = 128
    epoch_num = 3
    embedding_size = 200
    step = 0
    learning_rate = 0.001

    # 下载数据
    dataset_save_path = "./data/text8.txt"
    if not os.path.exists(dataset_save_path):
        dataset_download_path = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
        data_processor.download(save_path=dataset_save_path, corpus_url=dataset_download_path)
    # 获得数据集
    dataset, word2id_dict, id2word_dict = get_dataset(dataset_save_path, corpus_rate=0.2)
    data_loader = data_processor.build_batch(dataset, batch_size, epoch_num)
    # 初始化word2vec实例
    vocab_size = len(word2id_dict.keys())
    skip_gram = word2vec.SkipGram(vocab_size, embedding_size)
    # 开始模型训练
    train(skip_gram, data_loader)