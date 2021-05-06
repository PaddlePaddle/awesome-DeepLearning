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
from model import sentiment_classifier
from utils import data_processor


# 串联数据处理部分的以上环节，生成数据集dataset
def get_dataset(data_path, is_training=True):
    # 加载数据
    corpus = data_processor.load_imdb(data_path, is_training=is_training)
    # 对语料进行预处理
    corpus = data_processor.data_preprocess(corpus)
    # 根据语料构造字典，统计每个词的频率，并根据频率将每个词转换为一个整数id
    word2id_freq, word2id_dict = data_processor.build_dict(corpus)
    # 将语料转换为ID序列
    corpus = data_processor.convert_corpus_to_id(corpus, word2id_dict)

    return corpus, word2id_dict

def train(model, train_loader):
    model.train()

    # 判断可用的模型训练环境，优先使用GPU
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    # 创建优化器Optimizer，用于更新这个网络的参数
    optimizer = paddle.optimizer.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, parameters= model.parameters())

    # 开始训练
    for step, (sentences, labels) in enumerate(train_loader):
        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)
        pred, loss = model(sentences_var, labels_var)

        # 后向传播
        loss.backward()
        # 最小化loss
        optimizer.step()
        # 清除梯度
        optimizer.clear_grad()

        if step % 100 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))


if __name__ == '__main__':
    # 开始训练，定义一些训练过程中需要使用的超参数
    batch_size = 128
    epoch_num = 5
    embedding_size = 256
    learning_rate = 0.01
    max_seq_len = 128

    # 下载数据
    model_save_path = "./data/sentiment_classifiter.pdparams"
    dict_save_path = "./data/word.txt"
    dataset_save_path = "./data/aclImdb_v1.tar.gz"
    dataset_download_path = "https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz"
    if not os.path.exists(dataset_save_path):
        data_processor.download(save_path=dataset_save_path, corpus_url=dataset_download_path)

    # 加载数据集
    dataset, word2id_dict = get_dataset(dataset_save_path, is_training=True)

    data_loader = data_processor.build_batch(word2id_dict, dataset, batch_size, epoch_num, max_seq_len)

    # 初始化要训练的模型
    vocab_size = len(word2id_dict.keys())
    sentiment_classifier = sentiment_classifier.SentimentClassifier(embedding_size, vocab_size, num_steps=max_seq_len, num_layers=1)

    # 训练模型
    train(sentiment_classifier, data_loader)

    # 保存词典和模型
    with open(dict_save_path, "w") as f:
        for word_id, word in enumerate(word2id_dict.keys()):
            f.write(word+"\t"+str(word_id)+"\n")

    paddle.save(sentiment_classifier.state_dict(), model_save_path)