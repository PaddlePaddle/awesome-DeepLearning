import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from models.NNLM import NNLM
from utils import build_dataset,build_iterator


parser = argparse.ArgumentParser(description='骚话生成器')
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

def seqreate(str,vocab):
    """对测试输入做预处理：
          网络输入长度为5，不够PAD填充，长了截取最后5个
    """
    # str = '不得的时候'
    n_step = 5
    lin = str.strip()
    tokenizer = lambda x: [y for y in x]  # char-level
    words_line = []
    token = tokenizer(lin)

    # word to id
    if(len(token)<n_step):
        for i in range(n_step-len(token)):
            words_line.append(vocab.get('<PAD>'))
    for word in token:
        words_line.append(vocab.get(word, vocab.get('<UNK>')))

    return words_line[-n_step:]

def training(train_iter):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(10000):
        running_loss = 0
        for input_batch, target_batch in train_iter:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            output = model(input_batch)
            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%d' % (epoch + 1), 'cost =', '{:.6f}'.format(running_loss))

    torch.save(model, 'data/model.ckpt')

def testing(dataiter,vocab,string):
    """Test(文本生成)"""
    str = ''
    result = ''
    input = torch.LongTensor(seqreate(string, vocab)).to(device)   # 对测试输入做预处理
    for _ in input:
        str += number_dict[_.item()]   # 输入网络的真实字符串
    len_result = 0
    while True:
        len_result += 1
        model = torch.load('./data/model.ckpt')
        predict = model(input).data.max(1, keepdim=True)[1]
        str += number_dict[predict.item()]
        result += number_dict[predict.item()]
        input = torch.cat((input[1:], predict.reshape(1)))  # 结果作为网络的新输入，继续生成
        if len_result == 10:           # 直到生成句号标点，终止文本的生成
            break
    result = string + result
    print(f'生成: {result}')


if __name__ == "__main__":
    # 下面的目录、文件名按需更改。
    train_dir = "./data/train.txt"
    vocab_dir = "./data/vocab.pkl"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set, vocab, number_dict, nums_word = build_dataset(vocab_dir, train_dir)
    train_iter = build_iterator(train_set,batch_size=512)

    model = NNLM(nums_word).to(device)

    # training(train_iter)

    # string = '暴风雨'
    string = args.input
    print(f'输入: {string}')
    testing(train_iter,vocab,string)
