
# -*- coding: UTF-8 -*-

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
构建dataset，模型超参数配置，加载dataset，构建dataloader，加载预训练模型，设置AdamW优化器，cross entropy损失函数以及评估方式

'''

import yaml 
import argparse 
from pprint import pprint
from attrdict import AttrDict

import paddle
from paddle.io import DataLoader  

import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import TransformerModel, InferTransformerModel, CrossEntropyCriterion, position_encoding_init
from paddlenlp.utils.log import logger
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.transformers import ElectraForTokenClassification, ElectraTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict

import os
import pandas as pd
from sklearn.metrics import classification_report
from functools import partial


class TEDTalk(DatasetBuilder):

    SPLITS = {
        'train': 'train.tsv',
        'dev':'dev.tsv',
        'test': 'test.tsv'
    }

    def _get_data(self, mode, **kwargs):
        default_root='.'
        self.mode=mode
        filename = self.SPLITS[mode]
        fullname = os.path.join(default_root, filename)

        return fullname

    def _read(self, filename, *args):
        df=pd.read_csv(filename,sep='\t')
        for idx,row in df.iterrows():
            text=row['text_a']
            if(type(text)==float):
                print(text)
                continue
            tokens=row['text_a'].split()
            tags=row['label'].split()
            # if(self.mode=='train'):
            #     tags=row['label'].split()
            # else:
            #     tags = []
            yield {"tokens": tokens, "labels": tags}

    def get_labels(self):

        return ["0", "1", "2", "3"]
 
def load_dataset(path_or_read_func,
                 name=None,
                 data_files=None,
                 splits=None,
                 lazy=None,
                 **kwargs):
    
    reader_cls = TEDTalk
    print(reader_cls)
    if not name:
        reader_instance = reader_cls(lazy=lazy, **kwargs)
    else:
        reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

    datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
    return datasets
 
def tokenize_and_align_labels(example, tokenizer, no_entity_id,
                              max_seq_len=512):
    labels = example['labels']
    example = example['tokens']
    # print(labels)
    tokenized_input = tokenizer(
        example,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))
    # print(tokenized_input)
    return tokenized_input

def create_dataloader(args):
    '''
    构建dataset，模型超参数配置，加载dataset，构建dataloader，加载预训练模型，设置AdamW优化器，cross entropy损失函数以及评估方式
    return:
        train_data_loader
        test_data_loader
        model: 预训练模型
    '''

    # 加载dataset
    # Create dataset, tokenizer and dataloader.
    train_ds, test_ds = load_dataset('TEDTalk', splits=('train', 'test'), lazy=False)

    label_list = train_ds.label_list
    label_num = len(label_list)
    # no_entity_id = label_num - 1
    no_entity_id=0
    
    print(label_list)

    # 构建dataloader
    model_name_or_path='electra-base'
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

    trans_func = partial(
            tokenize_and_align_labels,
            tokenizer=tokenizer,
            no_entity_id=no_entity_id,
            max_seq_len=args.max_seq_length)
    train_ds = train_ds.map(trans_func)
 
    batchify_fn = lambda samples, fn=Dict({
            'input_ids': Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype='int32'),  # input
            'token_type_ids': Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype='int32'),  # segment
            'seq_len': Stack(dtype='int64'),  # seq_len
            'labels': Pad(axis=0, pad_val=args.ignore_label, dtype='int64')  # label
        }): fn(samples)
    
    train_batch_sampler = paddle.io.DistributedBatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    train_data_loader = DataLoader(
            dataset=train_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_sampler=train_batch_sampler,
            return_list=True) 

    test_ds = test_ds.map(trans_func)

    test_data_loader = DataLoader(
            dataset=test_ds,
            collate_fn=batchify_fn,
            num_workers=0,
            batch_size=args.batch_size,
            return_list=True)
    
    for index,data in enumerate(train_data_loader):
        # print(len(data))
        print(index)
        print(data)
        break

    return train_data_loader, test_data_loader  
 

      