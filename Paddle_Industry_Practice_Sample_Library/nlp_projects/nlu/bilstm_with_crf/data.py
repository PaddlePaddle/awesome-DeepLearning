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


import paddle
import json
from collections import OrderedDict
from paddle.io import Dataset
import numpy as np

class ATISDataset(Dataset):
    def __init__(self, path, vocab_path, intent_path, slot_path):
        self.examples = self.load_data(path)
        self.token2id, self.id2token = self.load_dict(vocab_path)
        self.intent2id, self.id2intent = self.load_dict(intent_path)
        self.slot2id, self.id2slot = self.load_dict(slot_path)


    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens, tags, intent = self.convert_example_to_id(example)
        
        return np.array(tokens), np.array(tags), intent, len(tokens)

    def __len__(self):
        return len(self.examples)

    @property
    def vocab_size(self):
        return len(self.token2id)

    @property
    def num_intents(self):
        return len(self.intent2id)

    @property
    def num_slots(self):
        return len(self.slot2id)


    def convert_example_to_id(self, example):
        tokens = example["text"].split()
        tags = example["tag"].split()
        intent = example["intent"]
        assert len(tokens) == len(tags)
        tokens = [self.token2id.get(token, "[unk]") for token in tokens]
        tags = [self.slot2id.get(tag, "O") for tag in tags]
        intent = self.intent2id[intent]

        return tokens, tags, intent

    def load_dict(self, dict_path):
        with open(dict_path, "r", encoding="utf-8") as f:
            words = [word.strip() for word in f.readlines()]
            dict2id = dict(zip(words, range(len(words))))
            id2dict = {v:k for k,v in dict2id.items()}

        return dict2id, id2dict



    def _split_with_id(self, text, start=0):
        word2sid = OrderedDict()
        word = ""
        count = 0
        for i in range(len(text)):
            if text[i] == " ":
                continue
            else:
                word += text[i]

            if (i < len(text) - 1 and text[i + 1] == " ") or i == len(text) - 1:
                # get whole word
                key = str(i - len(word) + 1 + start) + "_" + str(i + start) + "_" + word
                word2sid[key] = count
                count += 1
                word = ""
        return word2sid

    def load_data(self, path):
        examples = []
        raw_examples = []
        with open(path, "r", encoding="utf-8") as f:
            for example in f.readlines():
                raw_examples.append(json.loads(example))

        for raw_example in raw_examples:
            example = {}
            example["text"] = raw_example["text"]
            example["intent"] = raw_example["intent"]
            splited_text = raw_example["text"].split()
            tags = ['O'] * len(splited_text)
            word2sid = self._split_with_id(raw_example["text"])
            for entity in raw_example["entities"]:
                start, end, value, entity_name = entity["start"], entity["end"] - 1, entity["value"], entity["entity"]
                entity2sid = self._split_with_id(value, start=start)
                for i, word in enumerate(entity2sid.keys()):
                    if i == 0:
                        tags[word2sid[word]] = "B-" + entity_name
                    else:
                        tags[word2sid[word]] = "I-" + entity_name
            example["tag"] = " ".join(tags)
            examples.append(example)

        return examples
