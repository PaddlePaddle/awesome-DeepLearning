#这里在paddle上是能够正常运行的，这是提交的链接https://aistudio.baidu.com/aistudio/projectdetail/2259710
#使用了paddle上所配置的环境，所以直接运行的话应该跑不通

# 导入必要的库
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url

print("本项目基于Paddle的版本号为："+ paddle.__version__)


# AI Studio上的PaddleNLP版本过低，所以需要首先升级PaddleNLP
!pip install paddlenlp --upgrade


# 导入PaddleNLP相关的包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
# from utils import convert_example
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder

print("本项目基于PaddleNLP的版本号为："+ ppnlp.__version__)


MODEL_NAME = "ernie-1.0"
ernie_model = ppnlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)
model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)


# 定义ERNIE模型对应的 tokenizer，并查看效果
tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)


tokens = tokenizer._tokenize("王馨语学习笔记")
print("Tokens: {}".format(tokens))

# token映射为对应token id
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Tokens id: {}".format(tokens_ids))

# 拼接上预训练模型对应的特殊token ，如[CLS]、[SEP]
tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
print("Tokens id: {}".format(tokens_ids))
# 转化成paddle框架数据格式
tokens_pd = paddle.to_tensor([tokens_ids])
print("Tokens : {}".format(tokens_pd))

# 此时即可输入ERNIE模型中得到相应输出
sequence_output, pooled_output = ernie_model(tokens_pd)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))

encoded_text = tokenizer(text="王馨语学习笔记",  max_seq_len=20)
for key, value in encoded_text.items():
    print("{}:\n\t{}".format(key, value))

# 转化成paddle框架数据格式
input_ids = paddle.to_tensor([encoded_text['input_ids']])
print("input_ids : {}".format(input_ids))
segment_ids = paddle.to_tensor([encoded_text['token_type_ids']])
print("token_type_ids : {}".format(segment_ids))

# 此时即可输入 ERNIE 模型中得到相应输出
sequence_output, pooled_output = ernie_model(input_ids, segment_ids)
print("Token wise output: {}, Pooled output: {}".format(sequence_output.shape, pooled_output.shape))

!unzip -oq /home/aistudio/data/data78992/lcqmc.zip
# 删除解压后的无用文件
!rm -r __MACOSX

import pandas as pd

train_data = "./lcqmc/train.tsv"
train_data = pd.read_csv(train_data, header=None, sep='\t')
train_data.head(10)


class lcqmcfile(DatasetBuilder):
    SPLITS = {
        'train': 'lcqmc/train.tsv',
        'dev': 'lcqmc/dev.tsv',
    }

    def _get_data(self, mode, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            head = None
            for line in f:
                data = line.strip().split("\t")
                if not head:
                    head = data
                else:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}

    def get_labels(self):
        return ["0", "1"]


def load_dataset(name=None,
                 data_files=None,
                 splits=None,
                 lazy=None,
                 **kwargs):
    reader_cls = lcqmcfile
    print(reader_cls)
    if not name:
        reader_instance = reader_cls(lazy=lazy, **kwargs)
    else:
        reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

    datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
    return datasets

from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader

batch_size = 64
max_seq_length = 128

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),       # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")                               # label
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)


from paddlenlp.transformers import LinearDecayWithWarmup

# 训练过程中的最大学习率
learning_rate = 5e-5
# 训练轮次
epochs = 3
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()
metric = paddle.metric.Accuracy()



import paddle.nn.functional as F
from utils import evaluate

global_step = 0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
    evaluate(model, criterion, metric, dev_data_loader)



from utils import predict
import pandas as pd

label_map = {0:'0', 1:'1'}

def preprocess_prediction_data(data):
    examples = []
    for query, title in data:
        examples.append({"query": query, "title": title})
        # print(len(examples),': ',query,"---", title)
    return examples
test_file = 'lcqmc/test.tsv'
data = pd.read_csv(test_file, sep='\t')
# print(data.shape)
data1 = list(data.values)
examples = preprocess_prediction_data(data1)


results = predict(
        model, examples, tokenizer, label_map, batch_size=batch_size)

for idx, text in enumerate(examples):
    print('Data: {} \t Label: {}'.format(text, results[idx]))

data2 = []
for i in range(len(data1)):
    data2.extend(results[i])

data['label'] = data2
print(data.shape)
data.to_csv('lcqmc.tsv',sep='\t')