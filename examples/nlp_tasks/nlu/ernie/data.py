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


import json

def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        label_list = json.load(f)
    
    label2id = dict([(label, idx) for idx, label in enumerate(label_list)])
    id2label = dict([(idx, label) for label, idx in label2id.items()])

    return label2id, id2label

def read(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

        for example in examples:
            yield {"words":example[0], "slot_labels": example[1], "intent_labels":example[2], "history": example[4]}

def convert_example_to_feature(example, tokenizer, slot2id, intent2id,  use_history=False, pad_default_tag=0, max_seq_len=512):
    features = tokenizer(example["words"], is_split_into_words=True, max_seq_len=max_seq_len)
    # truncate slot sequence to make sure: the length of slot sequence is equal to word sequence
    slot_ids = [slot2id[slot] for slot in example["slot_labels"][:(max_seq_len-2)]]
    slot_ids = [slot2id[pad_default_tag]] + slot_ids + [slot2id[pad_default_tag]]
    assert len(features["input_ids"]) == len(slot_ids)

    # get intent feature
    intent_labels = [0] * len(intent2id)
    for intent in example["intent_labels"]:
        intent_labels[intent2id[intent]] = 1   

    # get history feature
    if use_history:
        history_features = tokenizer("".join(example["history"]), max_seq_len=max_seq_len)
    else:
        history_features = {"input_ids":[], "token_type_ids":[]}
    
    return features["input_ids"], features["token_type_ids"], intent_labels, slot_ids, history_features["input_ids"]

