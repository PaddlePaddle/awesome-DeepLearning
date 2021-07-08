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

import yaml 
import argparse 
from pprint import pprint
from attrdict import AttrDict

import xml.etree.ElementTree as ET 
from collections import Counter
import re

# 将测试集和训练集数据从xml格式提取成txt形式

if __name__ == '__main__': 
    # 读入参数
    yaml_file = './electra.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        # pprint(args)

    # 将xml形式数据解析，提取到字符串列表
    data_path = args.data_path
    file_path = data_path + args.dev_path
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(file_path, parser=xmlp)
    root = tree.getroot() 

    docs = []
    for doc_id in range(len(root[0])):
        doc_segs = []
        doc = root[0][doc_id]
        for seg in doc.iter('seg'): 
            doc_segs.append(seg.text)
        docs.extend(doc_segs)
    
    dev_texts = [re.sub(r'\s+', ' ', ''.join(d)).strip() for d in docs]
    with open(data_path + args.output_dev_path, 'w', encoding='utf-8') as f:
        for text in dev_texts:
            f.write(text + '\n')
    
    file_path = data_path + args.test_path

    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(file_path, parser=xmlp)
    root = tree.getroot()

    docs = []
    for doc_id in range(len(root[0])):
        doc_segs = []
        doc = root[0][doc_id]
        for seg in doc.iter('seg'):
            doc_segs.append(seg.text)
        docs.extend(doc_segs)

    test_texts_2012 = [re.sub(r'\s+', ' ', ''.join(d)).strip() for d in docs]

    # 将处理后的测试与训练数据【字符串列表形式】写入txt
    with open(data_path + args.output_test_path, 'w', encoding='utf-8') as f:
        for text in test_texts_2012:
            f.write(text + '\n')
    
    file_path = data_path + args.train_path
    with open(file_path) as f:
        xml = f.read()
    tree = ET.fromstring("<root>"+ xml + "</root>")
    
    docs = []
    for doc in tree.iter('transcript'):
        text_arr=doc.text.split('\n')
        text_arr=[item.strip() for item in text_arr if(len(item.strip())>2)]
        # print(text_arr)
        docs.extend(text_arr)
        # break

    train_texts=docs
    with open(data_path + args.output_train_path, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\n')



