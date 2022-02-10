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


import argparse
import paddle
from functools import partial
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddlenlp.datasets import load_dataset

from model import EventExtractionModel
from data import read, load_dict, load_schema, convert_example_to_features
from paddlenlp.data import Stack, Pad, Tuple
from seqeval.metrics.sequence_labeling import get_entities

def format_print(events):
    for idx, event in enumerate(events):
        print(f"event{idx} - event_type:{event['event_type']}, trigger:{event['trigger']}")
        for argument in event["arguments"]:
            print(f"role_type:{argument['role']}, argument:{argument['argument']} ")
        print()


def predict(trigger_model, role_model, tokenizer, trigger_id2tag, role_id2tag, schema):

    trigger_model.eval()
    role_model.eval()

    while True:
        input_text = input("input text: \n")
        if input_text == "quit":
            break
        splited_input_text = list(input_text.strip())
        features = tokenizer(splited_input_text, is_split_into_words=True, max_seq_len=args.max_seq_len, return_length=True)
        input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0)
        token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
        seq_len = features["seq_len"]
        
        trigger_logits = trigger_model(input_ids, token_type_ids)
        _, trigger_preds = trigger_model.viterbi_decoder(trigger_logits, paddle.to_tensor([seq_len]))
        trigger_preds = trigger_preds.numpy()[0][1:(seq_len-1)]
        trigger_preds = [trigger_id2tag[idx] for idx in trigger_preds]
        trigger_entities = get_entities(trigger_preds, suffix=False)
        
        role_logits = role_model(input_ids, token_type_ids)
        _, role_preds = role_model.viterbi_decoder(role_logits, paddle.to_tensor([seq_len]))
        role_preds = role_preds.numpy()[0][1:(seq_len-1)]
        role_preds = [role_id2tag[idx] for idx in role_preds]
        role_entities = get_entities(role_preds, suffix=False)

        events = []
        visited = set()
        for event_entity in trigger_entities:
            event_type, start, end = event_entity
            if event_type in visited:
                continue
            visited.add(event_type)
            events.append({"event_type":event_type, "trigger":"".join(splited_input_text[start:end+1]), "arguments":[]})
        
        for event in events:
            role_list = schema[event["event_type"]]
            for role_entity in role_entities:
                role_type, start, end = role_entity
                if role_type not in role_list:
                    continue
                event["arguments"].append({"role":role_type, "argument":"".join(splited_input_text[start:end+1])})
        
        format_print(events)    








if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger_model_path", type=str, default=None, help="trigger model path that you saved")
    parser.add_argument("--role_model_path", type=str, default=None, help="role model path that you saved")
    parser.add_argument("--trigger_tag_path", type=str, default=None, help="trigger dict path")
    parser.add_argument("--role_tag_path", type=str, default=None, help="role dict path")
    parser.add_argument("--schema_path", type=str, default=None, help="event schema path")
    parser.add_argument("--max_seq_len", type=int, default=512, help="max seq length")
    
    args = parser.parse_args()
    # yapf: enbale

    # load schema
    schema = load_schema(args.schema_path)
    
    # load dict
    model_name = "ernie-1.0"
    trigger_tag2id, trigger_id2tag = load_dict(args.trigger_tag_path)
    role_tag2id, role_id2tag = load_dict(args.role_tag_path)
    tokenizer = ErnieTokenizer.from_pretrained(model_name)

    # load model
    trigger_state_dict = paddle.load(args.trigger_model_path)
    role_state_dict = paddle.load(args.role_model_path)
    trigger_model = EventExtractionModel(ErnieModel.from_pretrained(model_name), num_classes=len(trigger_tag2id))
    role_model = EventExtractionModel(ErnieModel.from_pretrained(model_name), num_classes=len(role_tag2id))
    trigger_model.load_dict(trigger_state_dict)
    role_model.load_dict(role_state_dict)

    # predict
    predict(trigger_model, role_model, tokenizer, trigger_id2tag, role_id2tag, schema)

