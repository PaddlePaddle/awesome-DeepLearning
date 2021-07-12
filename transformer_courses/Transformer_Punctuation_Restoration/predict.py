
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

import os
import paddle
from paddlenlp.transformers import ElectraForTokenClassification, ElectraTokenizer

from dataloader import create_test_dataloader,load_dataset
from utils import evaluate, write2txt

def parse_decodes(input_words, id2label, decodes, lens):
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]

    outputs = []
    for idx, end in enumerate(lens):
        sent = input_words[idx]['tokens']
        tags = [id2label[x] for x in decodes[idx][1:end]]
        sent_out = []
        tags_out = []
        for s, t in zip(sent, tags):
            if(t=='0'):
                sent_out.append(s)
            else:
                # sent_out.append(s)
                sent_out.append(s+punctuation_dec[t])
        sent=' '.join(sent_out)
        sent=sent.replace(' ##','')
        outputs.append(sent)
    return outputs

def do_predict(test_data_loader):
    for step, batch in enumerate(test_data_loader):
        input_ids, token_type_ids, length, labels = batch
        logits = model(input_ids, token_type_ids)
        pred = paddle.argmax(logits, axis=-1)
        pred_list.append(pred.numpy())
        len_list.append(length.numpy())
    preds = parse_decodes(raw_data, id2label, pred_list, len_list)
    return preds

if __name__ == '__main__':
    # 读入参数
    yaml_file = './electra.base.yaml'
    with open(yaml_file, 'rt') as f:
        args = AttrDict(yaml.safe_load(f))
        # pprint(args)

    # 加载模型参数
    best_model = args.best_model
    init_checkpoint_path=os.path.join(args.output_dir, best_model)
    model_dict = paddle.load(init_checkpoint_path)

    # 加载dataset
    # Create dataset, tokenizer and dataloader.
    test_ds = load_dataset('TEDTalk', splits=('test'), lazy=False)
    label_list = test_ds.label_list
    label_num = len(label_list)

    # 加载模型与模型参数
    model = ElectraForTokenClassification.from_pretrained(args.model_name_or_path, num_classes=label_num)    
    model.set_dict(model_dict) 
     
    # 构建符号解码字典
    punctuation_dec = {
            '0': 'O',
            '1': ',',
            '2': '.',
            '3': '?',
        }
 
    id2label = dict(enumerate(label_list))
    raw_data = test_ds.data
 
    model.eval()
    pred_list = []
    len_list = []

    # 加载测试集data loader
    test_data_loader  = create_test_dataloader(args)

    # 设置损失函数 - Cross Entropy  
    loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=args.ignore_label)

    # 对测试集评估
    evaluate(model, loss_fct, test_data_loader, label_num)

    # 开始预测测试数据
    preds = do_predict(test_data_loader)

    # 将预测结果解码成真实句子，写入到txt文件
    if args.isSavingPreds == 1:
        write2txt(args, preds)  
