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


import paddle
from model import sentiment_classifier
from utils import data_processor


# 串联数据处理部分的以上环节，生成数据集dataset
def get_dataset(data_path, dict_path, is_training=True):
    # 加载数据
    corpus = data_processor.load_imdb(data_path, is_training=is_training)
    # 对语料进行预处理
    corpus = data_processor.data_preprocess(corpus)
    # 加载训练过程中生成的词典
    word2id_dict = data_processor.load_dict(dict_path)
    # 将语料转换为ID序列
    corpus = data_processor.convert_corpus_to_id(corpus, word2id_dict)

    return corpus, word2id_dict


def evaluate(model, test_loader):
    model.eval()

    # 判断可用的机器环境，优先使用GPU
    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    # 这里我们需要记录模型预测结果的准确率
    # 对于二分类任务来说，准确率的计算公式为：
    # (true_positive + true_negative) /
    # (true_positive + true_negative + false_positive + false_negative)
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for sentences, labels in test_loader:

        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)

        # 获取模型对当前batch的输出结果
        pred, loss = model(sentences_var, labels_var)

        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()
        for i in range(len(pred)):
            if labels[i][0] == 1:
                if pred[i][1] > pred[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i][1] > pred[i][0]:
                    fp += 1
                else:
                    tn += 1

    # 输出最终评估的模型效果
    print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))


if __name__ == '__main__':
    # 开始测试，定义一些训练过程中需要使用过的超参数
    batch_size = 128
    epoch_num = 5
    embedding_size = 256
    learning_rate = 0.01
    max_seq_len = 128

    # 下载数据
    dataset_save_path = "./data/aclImdb_v1.tar.gz"
    dict_save_path = "./data/word.txt"
    model_save_path = "./data/sentiment_classifiter.pdparams"

    # 加载数据集
    dataset, word2id_dict = get_dataset(dataset_save_path, dict_save_path, is_training=False)
    data_loader = data_processor.build_batch(word2id_dict, dataset, batch_size, epoch_num, max_seq_len)

    # 初始化并加载训练好的模型
    vocab_size = len(word2id_dict.keys())
    sentiment_classifier = sentiment_classifier.SentimentClassifier(embedding_size, vocab_size, num_steps=max_seq_len, num_layers=1)
    saved_state = paddle.load(model_save_path)
    sentiment_classifier.load_dict(saved_state)

    # 测试模型
    evaluate(sentiment_classifier, data_loader)
