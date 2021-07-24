import time
import os,re
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta


MAX_VOCAB_SIZE = 1000  # 字典表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
START = '<START>'


def build_vocab(file_path, max_size, min_freq):
    """建立字典"""
    word_dict = {}
    number_dict = {}
    tokenizer = lambda x: [y for y in x]  # char-level
    regEx = re.compile('[\\W]+')  # 规则是除单词，数字，下划线外的任意字符串去除
    # regEx = re.compile('')  # 规则是保留任意字符（有标点）
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):          # 按行读取
            line = regEx.split(line)  # 按字拆分
            for lin in line:
                lin = lin.strip()     # 去除头尾的空格或换行符
                if not lin:
                    continue
                for word in tokenizer(lin):
                    word_dict[word] = word_dict.get(word, 0) + 1  # 记录文本中的字及其出现次数
        # vocab_list 按字的出现频次由大到小排序
        vocab_list = sorted([_ for _ in word_dict.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        # 给字典中的字绑定索引号
        word_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        number_dict = {idx: word_count[0] for idx, word_count in enumerate(vocab_list)}
        # <UNK>、<PAD>字符加入字典中
        number_dict.update({len(word_dict): UNK, (len(word_dict) + 1): PAD, (len(word_dict) + 2): START})
        word_dict.update({UNK: len(word_dict), PAD: len(word_dict) + 1,  START: len(word_dict) + 2})
    return word_dict,number_dict


def build_dataset(vocab_path,train_path):
    # 根据语料新建字典
    vocab,number_dict = build_vocab(train_path, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(vocab, open(vocab_path, 'wb'))
    # print(f"Vocab size: {len(vocab)}")

    def Gram(sequence,t):
        "获取当前时刻的上文, 5-context"
        t1 = sequence[t-1] if t-1>=0 else vocab.get(PAD)  # 上文不存在用PAD填充
        t2 = sequence[t-2] if t-2>=0 else vocab.get(PAD)
        t3 = sequence[t-3] if t-3>=0 else vocab.get(PAD)
        t4 = sequence[t - 4] if t - 4 >= 0 else vocab.get(PAD)
        t5 = sequence[t - 5] if t - 5 >= 0 else vocab.get(PAD)
        return [t5,t4,t3,t2,t1]

    def load_dataset(path):
        """获取上下文对"""
        dataset = []
        tokenizer = lambda x: [y for y in x]  # char-level
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                words_line = []
                token = tokenizer(lin)

                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                for i in range(3, len(words_line)):  # 注意第一条上下文对在哪里
                    input = Gram(words_line, i)
                    input.append(words_line[i])
                    target = input[1:]
                    input[0] = vocab.get(START)
                    dataset.append((input[:-1], target))
        return dataset

    trainset = load_dataset(train_path)
    return trainset,vocab,number_dict,len(vocab)


class DatasetIterater(object):
    """对数据集进行batch拆分"""
    def __init__(self, batches, batch_size):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas])
        y = torch.LongTensor([_[1] for _ in datas])
        return x,y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, batch_size):
    iter = DatasetIterater(dataset, batch_size)
    return iter

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax

# if __name__ == "__main__":
#     # 下面的目录、文件名按需更改。
#     train_dir = "./data/train.txt"
#     vocab_dir = "./data/vocab.pkl"
#
#     data_set,vocab,number_dict,nums_word = build_dataset(vocab_dir,train_dir)
#     iter = build_iterator(data_set,batch_size=32)




