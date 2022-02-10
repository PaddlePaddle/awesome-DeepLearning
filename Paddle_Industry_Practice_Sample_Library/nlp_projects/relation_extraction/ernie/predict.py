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
import json
import argparse
from functools import partial
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from utils.metric import SPOMetric
from utils.utils import decoding
from model import ErnieForTokenClassification
from data import read, generate_dict, load_dict, load_schema, load_reverse_schema, convert_example_to_feature


def predict(model, tokenizer, reverse_schemas, id2label):

    model.eval()
    while True:
        input_text = input("input text: \n")
        if not input_text:
            continue
        if input_text == "quit":
            break

        # processing input text
        splited_input_text = list(input_text)
        features = tokenizer(splited_input_text, is_split_into_words=True, max_seq_len=args.max_seq_len, return_length=True)
        input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0)
        token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
        seq_lens = paddle.to_tensor([features["seq_len"]])

        # predict by model and decoding result 
        logits = model(input_ids, token_type_ids=token_type_ids)
        examples = [{"text":input_text}]
        pred_examples = decoding(examples, reverse_schema, logits, seq_lens, id2label)

        # print pred_examples
        print(json.dumps(pred_examples, indent=4, ensure_ascii=False))
    


if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--ori_schema_path", type=str, default=None, help="schema path")
    parser.add_argument("--save_label_path", type=str, default=None, help="schema path")
    parser.add_argument("--save_schema_path", type=str, default=None, help="reverse schema path")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # generate label and schema dict
    generate_dict(args.ori_schema_path, args.save_label_path, args.save_schema_path)

    # load dev data
    model_name = "ernie-1.0"
    label2id, id2label = load_dict(args.save_label_path)
    reverse_schema = load_reverse_schema(args.save_schema_path)

    tokenizer = ErnieTokenizer.from_pretrained(model_name)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    model = ErnieForTokenClassification(ernie, num_classes=len(label2id))    
    model.load_dict(loaded_state_dict)

    # predicting
    predict(model, tokenizer, reverse_schema, id2label)
    
