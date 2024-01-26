import argparse
import numpy as np
from models.RNN import RNN
from utils import build_dataset,build_iterator


def seqreate(str,vocab):
    """对测试输入做预处理：
          网络输入长度为5，不够PAD填充，长了截取最后5个
    """
    n_step = 5
    lin = str.strip()
    tokenizer = lambda x: [y for y in x]  # char-level
    words_line = []
    token = tokenizer(lin)

    # # word to id
    # if(len(token)<n_step):
    #     for i in range(n_step-len(token)):
    #         words_line.append(vocab.get('<PAD>'))
    for word in token:
        words_line.append(vocab.get(word, vocab.get('<UNK>')))

    return words_line

def testing(model,vocab,string):
    """Test(文本生成)"""
    str = ''
    input = seqreate(string, vocab)  # 对测试输入做预处理
    result = seqreate(string, vocab)
    input.insert(0,vocab.get('<START>'))
    for _ in input:
        str += number_dict[_]   # 输入网络的真实字符串
    print(f'输入: {str}')

    o, s = model.forward_propagation(input)
    # 取出对每个word后的正确word的估计
    o = o.tolist()
    predict = o[-1].index(max(o[-1]))
    result.append(predict)

    str += number_dict[predict]
    result_str = ' '
    for _ in result:
        result_str += number_dict[_]  # 输入网络的真实字符串
    print(f'生成: {result_str}')

if __name__ == '__main__':
    # 下面的目录、文件名按需更改。
    train_dir = "./data/train.txt"
    vocab_dir = "./data/vocab.pkl"

    train_set, vocab, number_dict, nums_word = build_dataset(vocab_dir, train_dir)
    train_iter = build_iterator(train_set,batch_size=32)

    model = RNN(473,100,4)
    model.train_with_sgd(train_iter)

    string = '暴风'
    # print(f'输入: {string}')
    testing(model, vocab, string)