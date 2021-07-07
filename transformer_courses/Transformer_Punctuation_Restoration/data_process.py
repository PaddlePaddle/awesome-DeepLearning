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
预处理数据，并且构建数据集
'''
import yaml 
import argparse 
from pprint import pprint
from attrdict import AttrDict

from tqdm import tqdm 
import ujson
import codecs
import os
import re
import pandas as pd 

from paddlenlp.transformers import ElectraForTokenClassification, ElectraTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict

def clean_text(text):
    '''
    将符号替换为
    '''
    text = text.replace('!', '.')
    text = text.replace(':', ',')
    text = text.replace('--', ',')
    
    reg = "(?<=[a-zA-Z])-(?=[a-zA-Z]{2,})"
    r = re.compile(reg, re.DOTALL)
    text = r.sub(' ', text)
    
    text = re.sub(r'\s-\s', ' , ', text)
    
#     text = text.replace('-', ',')
    text = text.replace(';', '.')
    text = text.replace(' ,', ',')
    text = text.replace('♫', '')
    text = text.replace('...', '')
    text = text.replace('.\"', ',')
    text = text.replace('"', ',')

    text = re.sub(r'--\s?--', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r',\s?,', ',', text)
    text = re.sub(r',\s?\.', '.', text)
    text = re.sub(r'\?\s?\.', '?', text)
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\s+\?', '?', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\.[\s+\.]+', '. ', text)
    text = re.sub(r'\s+\.', '.', text)
    
    return text.strip().lower()

def format_data(train_text):
    '''
    依据文本中出现的符号，分别生成文本tokens以及对应标签
    return:
        texts：文本tokens列表，每一个item是一个文本样本对应的tokens列表
        labels：标点符号标签列表，每一个item是一个标点符号标签列表，代表token的下一个位置的标点符号
    '''
    labels=[]
    texts=[]
    for line in tqdm(train_text):
        line=line.strip()
        if(len(line)==2):
            print(line)
            continue
        text=tokenizer.tokenize(line)
        label=[]
        cur_text=[]
        flag=True
        for item in text:
            if(item in punctuation_enc):
                # print(item)
                if(len(label)>0):
                    label.pop()
                    label.append(punctuation_enc[item])
                else:
                    print(text)
                    falg=False
                    break
            else:
                cur_text.append(item)
                label.append(punctuation_enc['O'])
        if(flag):
            labels.append(label)
            texts.append(cur_text)
    return texts,labels

def write_json(filename, dataset):
    print('write to'+filename)
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)

if __name__ == '__main__': 
    # 读入参数
    yaml_file = './electra.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        # pprint(args)

    # 数据读取
    with open(args.data_path + args.output_train_path, 'r', encoding='utf-8') as f:
        train_text = f.readlines()
    with open(args.data_path + args.output_dev_path, 'r', encoding='utf-8') as f:
        valid_text = f.readlines()
    with open(args.data_path + args.output_test_path, 'r', encoding='utf-8') as f:
        test_text = f.readlines()
    
    datasets = train_text, valid_text, test_text
    
    datasets = [[clean_text(text) for text in ds] for ds in datasets]

    # 利用electra的分词工具进行分词，然后构造数据集

    model_name_or_path=args.model_name_or_path
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

    punctuation_enc = {
            'O': '0',
            ',': '1',
            '.': '2',
            '?': '3',
        }
    
    # 以一个文本序列为例，引入 format_data 函数的构建
    text=tokenizer.tokenize("all the projections [ say that ] this one [ billion ] will [ only ] grow with one to two or three percent")
    print(text)
    label=[]
    cur_text=[]
    for item in text:
        if(item in punctuation_enc):
            print(item)
            label.pop()
            label.append(punctuation_enc[item])
        else:
            cur_text.append(item)
            label.append(punctuation_enc['O'])
    # label=[item for item in text]
    print(label)
    print(cur_text)
    print(len(label))
    print(len(cur_text))

 
    # 构建训练集
    texts,labels=format_data(train_text)

    print(len(texts))
    print(texts[0])
    print(labels[0])

    def output_to_tsv(texts,labels,file_name):
        data=[]
        for text,label in zip(texts,labels):
            if(len(text)!=len(label)):
                print(text)
                print(label)
                continue
            data.append([' '.join(text),' '.join(label)])
        df=pd.DataFrame(data,columns=['text_a','label'])
        df.to_csv(file_name,index=False,sep='\t')
    
    def output_to_train_tsv(texts,labels,file_name):
        data=[]
        for text,label in zip(texts,labels):
            if(len(text)!=len(label)):
                print(text)
                print(label)
                continue
            if(len(text)==0):
                continue
            data.append([' '.join(text),' '.join(label)])
        # data=data[65000:70000]
        df=pd.DataFrame(data,columns=['text_a','label'])
        df.to_csv(file_name,index=False,sep='\t')

    output_to_train_tsv(texts,labels, args.output_train_tsv)

    texts,labels=format_data(test_text)
    output_to_tsv(texts,labels, args.output_test_tsv)
    texts,labels=format_data(valid_text)
    output_to_tsv(texts,labels, args.output_dev_tsv)

    raw_path='.'
    train_file = os.path.join(raw_path, args.output_train_tsv)
    dev_file = os.path.join(raw_path, args.output_dev_tsv)
    
    train_data=pd.read_csv(train_file,sep='\t')
    train_data.head()
