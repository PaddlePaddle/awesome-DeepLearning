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

import copy
import argparse
from tqdm import tqdm
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from utils.metric import SPOMetric
from utils.utils import decoding
from model import ErnieForTokenClassification
from data import read, generate_dict, load_dict, load_schema, load_reverse_schema, convert_example_to_feature


def collate_fn(batch, pad_default_token_id=0):
    input_ids_list, token_type_ids_list, attention_mask_list, seq_len_list, labels_list = [], [], [], [], []
    for input_ids, token_type_ids, attention_mask, seq_len, labels in batch:
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        seq_len_list.append(seq_len)
        labels_list.append(labels)

    # padding sequence
    max_len = max(seq_len_list)
    for idx in range(len(input_ids_list)):
        pad_len = max_len - seq_len_list[idx]
        input_ids_list[idx] = input_ids_list[idx] + [pad_default_token_id] * pad_len
        token_type_ids_list[idx] = token_type_ids_list[idx] + [0] * pad_len
        attention_mask_list[idx] = attention_mask_list[idx] + [0] * pad_len

        pad_label = labels_list[idx][0][:] # CLS label
        labels_list[idx] = labels_list[idx] + [pad_label] * pad_len


    return paddle.to_tensor(input_ids_list), paddle.to_tensor(token_type_ids_list), paddle.to_tensor(attention_mask_list), paddle.to_tensor(seq_len_list), paddle.to_tensor(labels_list)


def evaluate(model, data_loader, metric, examples, reverse_schemas, id2label, batch_size):

    model.eval()
    metric.reset()
    for idx, batch_data in tqdm(enumerate(data_loader)):
        input_ids, token_type_ids, attention_masks, seq_lens, labels = batch_data
        logits = model(input_ids, token_type_ids=token_type_ids)        
        # decoding logits into examples with spo_list
        batch_examples = examples[idx*batch_size : (idx+1)*batch_size]
        batch_pred_examples = decoding(batch_examples, reverse_schemas, logits, seq_lens, id2label)
        # count metric
        metric.update(batch_examples, batch_pred_examples)


    precision, recall, f1 = metric.accumulate()

    return precision, recall, f1


if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--test_path", type=str, default=None, help="test data")
    parser.add_argument("--ori_schema_path", type=str, default=None, help="schema path")
    parser.add_argument("--save_label_path", type=str, default=None, help="schema path")
    parser.add_argument("--save_schema_path", type=str, default=None, help="reverse schema path")
    parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # generate label and schema dict
    generate_dict(args.ori_schema_path, args.save_label_path, args.save_schema_path)

    # load dev data
    model_name = "ernie-1.0"
    label2id, id2label = load_dict(args.save_label_path)
    reverse_schema = load_reverse_schema(args.save_schema_path)
    
    test_ds = load_dataset(read, data_path=args.test_path, lazy=False)
    examples = copy.deepcopy(test_ds)    
    
    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id,  pad_default_label="O", max_seq_len=args.max_seq_len)
    test_ds = test_ds.map(trans_func, lazy=False)

    # Warning: you should not set shuffle to True
    batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = paddle.io.DataLoader(test_ds, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    model = ErnieForTokenClassification(ernie, num_classes=len(label2id))    
    model.load_dict(loaded_state_dict)

    metric = SPOMetric()
 
    # evalute on dev data
    precision, recall, f1  = evaluate(model, test_loader,  metric, examples, reverse_schema, id2label, args.batch_size)
    print(f'evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')
    
