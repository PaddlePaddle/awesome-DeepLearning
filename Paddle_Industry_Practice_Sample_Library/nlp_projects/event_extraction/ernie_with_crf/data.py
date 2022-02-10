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


import os
import json


def data_process(path, is_predict=False):
    def tagging_data(data, start, l, _type):
        """label_data"""
        for i in range(start, start + l):
            prefix = "B-" if i == start else "I-"
            try:
                data[i] = "{}{}".format(prefix, _type)
            except:
                print(start, l, data)
                exit(0)
        return data

    trigger_examples = ["text_a"] if is_predict else ["text_a\tlabel"]
    role_examples = ["text_a"] if is_predict else ["text_a\tlabel"]
    for line in read_by_lines(path):
        example = json.loads(line)
        text_a = [
            "，" if t == " " or t == "\n" or t == "\t" else t
            for t in list(example["text"].lower())
        ]
        if is_predict:
            trigger_examples.append("\002".join(text_a))
            role_examples.append("\002".join(text_a))
        else:
            # process trigger
            trigger_tags = ["O"] * len(text_a)
            for event in example.get("event_list", []):
                event_type, trigger_start_index, trigger = event[
                    "event_type"], event["trigger_start_index"], event[
                        "trigger"]
                trigger_tags = tagging_data(trigger_tags, trigger_start_index,
                                            len(trigger), event_type)
            trigger_examples.append("{}\t{}".format("\002".join(text_a),
                                                    "\002".join(trigger_tags)))

            # process role
            for event in example.get("event_list", []):
                role_tags = ["O"] * len(text_a)
                for arg in event["arguments"]:
                    role_type, argument_start_index, argument = arg[
                        "role"], arg["argument_start_index"], arg["argument"]
                    role_tags = tagging_data(role_tags, argument_start_index,
                                             len(argument), role_type)
                role_examples.append("{}\t{}".format("\002".join(text_a),
                                                     "\002".join(role_tags)))

    return trigger_examples, role_examples


def schema_process(schema_path):
    trigger_tags, role_tags = ["O"], ["O"]
    for line in read_by_lines(schema_path):
        event_schema = json.loads(line.strip())
        # process trigger
        event_type = event_schema["event_type"]
        if "B-{}".format(event_type) not in trigger_tags:
            trigger_tags.append("B-{}".format(event_type))
            trigger_tags.append("I-{}".format(event_type))
        # process role
        for role in event_schema["role_list"]:
            role_name = role["role"]
            if "B-{}".format(role_name) not in role_tags:
                role_tags.append("B-{}".format(role_name))
                role_tags.append("I-{}".format(role_name))

    return trigger_tags, role_tags


# load data by line from file
def read_by_lines(path):
    result = list()
    with open(path, "r") as infile:
        for line in infile:
            result.append(line.strip())
    return result


# write data to file
def write_by_lines(path, data):
    with open(path, "w") as outfile:
        [outfile.write(d + "\n") for d in data]


# load and process data to suitable format
def data_prepare(path):
    # schema process: process event_schema file to obtain trigger and role dictionary
    dict_path = os.path.join(path, "dict")
    schema_path = os.path.join(path, "duee_event_schema.json")
    if not os.path.exists(dict_path):
        os.mkdir(dict_path)
    trigger_tags, role_tags = schema_process(schema_path)
    write_by_lines(os.path.join(dict_path, "trigger.dict"), trigger_tags)
    write_by_lines(os.path.join(dict_path, "role.dict"), role_tags)

    # data process: process original DuEE Dataset to suitable format (sequence labeling format)
    trigger_path = os.path.join(path, "trigger")
    role_path = os.path.join(path, "role")
    if not os.path.exists(trigger_path):
        os.mkdir(trigger_path)
    if not os.path.exists(role_path):
        os.mkdir(role_path)

    train_trigger, train_role = data_process(
        os.path.join(path, "duee_train.json"))
    dev_trigger, dev_role = data_process(os.path.join(path, "duee_dev.json"))
    test_trigger, test_role = data_process(os.path.join(
        path, "duee_test.json"))
    write_by_lines(os.path.join(trigger_path, "duee_train.tsv"), train_trigger)
    write_by_lines(os.path.join(trigger_path, "duee_dev.tsv"), dev_trigger)
    write_by_lines(os.path.join(trigger_path, "duee_test.tsv"), test_trigger)
    write_by_lines(os.path.join(role_path, "duee_train.tsv"), train_role)
    write_by_lines(os.path.join(role_path, "duee_dev.tsv"), dev_role)
    write_by_lines(os.path.join(role_path, "duee_test.tsv"), test_role)


# load trigger and role dict
def load_dict(dict_path):
    tag2id, id2tag = {}, {}
    with open(dict_path, "r", encoding="utf-8") as f:
       for idx, line in enumerate(f.readlines()):
           word = line.strip()
           id2tag[idx] = word
           tag2id[word] = idx
    
    return tag2id, id2tag

# load schema file
def load_schema(schema_path):
    schema = {}
    with open(schema_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
           event_des = json.loads(line)
           schema[event_des["event_type"]] = [r["role"] for r in event_des["role_list"]]
    return schema

# load data from local file, which will be used for loading data with paddlenlp
def read(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f.readlines():
            words, labels = line.strip().split("\t")
            words = words.split("\002")
            labels = labels.split("\002")
            yield {"tokens": words, "labels":labels}


def convert_example_to_features(example, tokenizer, tag2id, max_seq_length=512, pad_default_tag="O",  is_test=False):
   
    features = tokenizer(example["tokens"], is_split_into_words=True,  max_seq_len=max_seq_length, return_length=True)
    if is_test:
        return features["input_ids"], features["token_type_ids"], features["seq_len"]

    tag_ids = [tag2id[tag] for tag in example["labels"][:(max_seq_length-2)]]
    tag_ids = [tag2id[pad_default_tag]] + tag_ids + [tag2id[pad_default_tag]]
    assert len(features["input_ids"]) == len(tag_ids)
    
    return features["input_ids"], features["token_type_ids"],  features["seq_len"], tag_ids  
