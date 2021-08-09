import re
import os
import time
import tarfile
import random
import argparse
import numpy as np
from functools import partial

import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.metric import Accuracy


class IMDBDataset(Dataset):
    def __init__(self, is_training=True):
        self.data = self.load_imdb(is_training)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


    def load_imdb(self, is_training):
        # 将读取的数据放到列表data_set里
        data_set = []

        # data_set中每个元素都是一个二元组：(句子，label)，其中label=0表示消极情感，label=1表示积极情感
        for label in ["pos", "neg"]:
            with tarfile.open("./imdb_aclImdb_v1.tar.gz") as tarf:
                path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                    else "aclImdb/test/" + label + "/.*\.txt$"
                path_pattern = re.compile(path_pattern)
                tf = tarf.next()
                while tf != None:
                    if bool(path_pattern.match(tf.name)):
                        sentence = tarf.extractfile(tf).read().decode()
                        sentence_label = 0 if label == 'neg' else 1
                        data_set.append({"sentence":sentence, "label":sentence_label}) 
                    tf = tarf.next()

        return data_set


def convert_example(example,
                    tokenizer,
                    label_list,
                    max_seq_length=512,
                    is_test=False):
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example['label']
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    if (int(is_test) + len(example)) == 2:
        example = tokenizer(
            example['sentence'],
            max_seq_len=max_seq_length,
            return_attention_mask=True)
    else:
        example = tokenizer(
            example['sentence1'],
            text_pair=example['sentence2'],
            max_seq_len=max_seq_length,
            return_attention_mask=True)

    if not is_test:
        return example['input_ids'], example['token_type_ids'], example[
            'attention_mask'], label
    else:
        return example['input_ids'], example['token_type_ids'], example[
            'attention_mask']

