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
from utils.extract_chinese_and_punctuation import ChineseAndPunctuationExtractor


def generate_dict(ori_file_path, save_label_path, save_schema_path):
    predicates, schemas = ["O", "I"], {}
    # generate schema dict
    with open(ori_file_path, "r", encoding="utf-8") as f:
        for relation_schema in f.readlines():
            relation_schema = json.loads(relation_schema)
            predicate, subject_type, object_types  = relation_schema["predicate"], relation_schema["subject_type"], relation_schema["object_type"]
            schemas[predicate] = {"object_type": object_types, "subject_type": subject_type}

            predicates.append("S-"+predicate+"-"+subject_type)
            for object_type in object_types:
                predicates.append("O-"+predicate+"-"+object_types[object_type])

    with open(save_label_path, "w", encoding="utf-8") as f:
        for predicate in predicates:
            f.write(predicate+"\n")
    print(f"predicate dict has saved: {save_label_path}")

    with open(save_schema_path, "w", encoding="utf-8") as f:
        json.dump(schemas, f, ensure_ascii=False, indent=4)
    print(f"schema file has saved: {save_schema_path}")



def get_object_keyname(reverse_schema, predicate, object_valname):
    object_type_keyname = reverse_schema[predicate][object_valname]
    return object_type_keyname


def load_schema(schema_path):
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    return schema

def load_reverse_schema(schema_path):
    schemas = load_schema(schema_path)
    reverse_schemas = copy.deepcopy(schemas)

    for reverse_schema in reverse_schemas:
        object_type = reverse_schemas[reverse_schema]["object_type"]
        reverse_schemas[reverse_schema]["object_type"] = dict([(v,k) for k,v in object_type.items()])
    
    return reverse_schemas
            

def load_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word2id = dict(zip(words, range(len(words))))
        id2word = dict(zip(range(len(words)), words))
    
    return word2id, id2word


def read(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        for example in f.readlines():
            example = json.loads(example)
            yield example
        
         
def find_entity(input_ids, entity_ids):
    entity_len = len(entity_ids)
    match_start, match_end = -1, -1
    for idx in range(len(input_ids)):
        if input_ids[idx:idx+entity_len] == entity_ids:
            match_start = idx
            match_end = idx+entity_len-1
            break

    return match_start, match_end


def find_entity_with_visited(input_ids, entity_ids, visited):
    entity_len = len(entity_ids)
    match_start, match_end = -1, -1
    for idx in range(len(input_ids)):
        if sum(visited[idx:idx+entity_len])!=0:
            continue
        if input_ids[idx:idx+entity_len] == entity_ids:
            match_start = idx
            match_end = idx+entity_len-1
            break
    return match_start, match_end


def convert_example_to_feature1(example, label2id, tokenizer, pad_default_label="O", max_seq_len=512):
    # convert word sequence to feature
    features = tokenizer(list(example["text"]), is_split_into_words=True, max_seq_len=max_seq_len, return_length=True, return_attention_mask=True)
    input_ids, token_type_ids, attention_mask, seq_len = features["input_ids"], features["token_type_ids"], features["attention_mask"], features["seq_len"]
    # construct labels
    labels = [[0] * len(label2id) for _ in range(seq_len)]
    
    spo_list = example["spo_list"]  if "spo_list" in example.keys() else []
    for spo in spo_list:
        subject_label = "S-"+spo["predicate"]+"-"+spo["subject_type"]
        subject_ids = tokenizer.convert_tokens_to_ids(list(spo["subject"]))
        entities = [(subject_label, subject_ids)]
        for object_type in spo["object_type"]:
            object_label = "O-"+spo["predicate"]+"-"+spo["object_type"][object_type]
            object_ids = tokenizer.convert_tokens_to_ids(list(spo["object"][object_type]))
            entities.append((object_label, object_ids))
        
        visited = [0] * seq_len
        entities = sorted(entities, key=lambda entity: len(entity[1]), reverse=True)
        for entity in entities:
            entity_label, entity_ids = entity
       
            match_start, match_end = find_entity_with_visited(input_ids, entity_ids, visited)
            if match_start < 0:
                match_start, match_end = find_entity(input_ids, entity_ids)
            assert match_start >= 0
            for idx in range(match_start, match_end+1):
                visited[idx] = 1
                labels[idx][label2id[entity_label]] = 1
    
    for idx in range(seq_len):
        if sum(labels[idx]) == 0:
            labels[idx][0] = 1

    return input_ids, token_type_ids, attention_mask, seq_len, labels
                           
            
def convert_example_to_feature2(example, label2id, tokenizer, pad_default_label="O", max_seq_len=512):
    # convert word sequence to feature
    features = tokenizer(list(example["text"]), is_split_into_words=True, max_seq_len=max_seq_len, return_length=True, return_attention_mask=True)
    input_ids, token_type_ids, attention_mask, seq_len = features["input_ids"], features["token_type_ids"], features["attention_mask"], features["seq_len"]
    # construct labels
    labels = [[0] * len(label2id) for _ in range(seq_len)]

    spo_list = example["spo_list"]  if "spo_list" in example.keys() else []
    for spo in spo_list:
        subject_label = "S-"+spo["predicate"]+"-"+spo["subject_type"]
        subject_ids = tokenizer.convert_tokens_to_ids(list(spo["subject"]))
        entities = [(subject_label, subject_ids)]
        for object_type in spo["object_type"]:
            object_label = "O-"+spo["predicate"]+"-"+spo["object_type"][object_type]
            object_ids = tokenizer.convert_tokens_to_ids(list(spo["object"][object_type]))
            entities.append((object_label, object_ids))

        visited = [0] * seq_len
        entities = sorted(entities, key=lambda entity: len(entity[1]), reverse=True)
        for entity in entities:
            entity_label, entity_ids = entity
            match_start, match_end = find_entity_with_visited(input_ids, entity_ids, visited)
            if match_start < 0:
                match_start, match_end = find_entity(input_ids, entity_ids)
            assert match_start >= 0
            for i, idx in enumerate(range(match_start, match_end+1)):
                visited[idx] = 1
                if i == 0:
                    labels[idx][label2id[entity_label]] = 1
                else:
                    labels[idx][1] = 1

    for idx in range(seq_len):
        if sum(labels[idx]) == 0:
            labels[idx][0] = 1

    return input_ids, token_type_ids, attention_mask, seq_len, labels        


def convert_example_to_feature(example, label2id, tokenizer, pad_default_label="O", max_seq_len=512):
    # convert word sequence to feature
    features = tokenizer(list(example["text"]), is_split_into_words=True, max_seq_len=max_seq_len, return_length=True, return_attention_mask=True)
    input_ids, token_type_ids, attention_mask, seq_len = features["input_ids"], features["token_type_ids"], features["attention_mask"], features["seq_len"]
    # construct labels
    labels = [[0] * len(label2id) for _ in range(seq_len)]

    spo_list = example["spo_list"]  if "spo_list" in example.keys() else []
    for spo in spo_list:
        subject_label = "S-"+spo["predicate"]+"-"+spo["subject_type"]
        subject_ids = tokenizer.convert_tokens_to_ids(list(spo["subject"]))
        entities = [(subject_label, subject_ids)]
        for object_type in spo["object_type"]:
            object_label = "O-"+spo["predicate"]+"-"+spo["object_type"][object_type]
            object_ids = tokenizer.convert_tokens_to_ids(list(spo["object"][object_type]))
            entities.append((object_label, object_ids))

        visited = [0] * seq_len
        entities = sorted(entities, key=lambda entity: len(entity[1]), reverse=True)
        for entity in entities:
            entity_label, entity_ids = entity
            match_start, match_end = find_entity_with_visited(input_ids, entity_ids, visited)
            
            if match_start >= 0:
                for i, idx in enumerate(range(match_start, match_end+1)):
                    visited[idx] = 1
                    if i == 0:
                        labels[idx][label2id[entity_label]] = 1
                    else:
                        labels[idx][1] = 1

    for idx in range(seq_len):
        if sum(labels[idx]) == 0:
            labels[idx][0] = 1

    return input_ids, token_type_ids, attention_mask, seq_len, labels 
