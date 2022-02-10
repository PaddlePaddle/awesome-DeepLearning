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
import paddle
import random
import numpy as np
from collections import defaultdict

def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_object_keyname(reverse_schema, predicate, object_valname):
    object_type_keyname = reverse_schema[predicate]["object_type"][object_valname]
    return object_type_keyname


def _decoding_by_label(label_logits):
    labels = []
    i, lens = 0, len(label_logits)
    last_yes = False
    while i < lens:
        if label_logits[i] == 0:
            last_yes=False
        else:
            if last_yes:
                if i==0:
                    labels.append([])
                labels[-1].append(i)
            else:
                labels.append([])
                labels[-1].append(i)
            last_yes=True
        i += 1

    return labels
   
     
def parsing_entity(label_logits, label_I_logits):
    token_sids = np.argwhere(label_logits==1).squeeze(-1)
    extract_results = []
    for token_sid in token_sids:
        extract_result = [token_sid]
        cursor = token_sid + 1
        while cursor<len(label_I_logits) and label_I_logits[cursor] == 1:
            extract_result.append(cursor)
            cursor += 1
        extract_results.append(extract_result)

    return extract_results


def decoding(examples, reverse_schema, batch_logits, batch_seq_len, id2label):
    assert len(examples) == len(batch_logits)
    batch_pred_examples = []
    for example_id, (logits, seq_len) in enumerate(zip(batch_logits, batch_seq_len)):
        spo_list, extract_subjects, extract_objects = [], defaultdict(list), defaultdict(list)
        example_text = examples[example_id]["text"]
        logits = logits[1:(seq_len-1)]
        logits[logits>=0.5] = 1
        logits[logits<0.5] = 0

        assert len(logits) == len(example_text)

        # get label I logits
        label_I_logits = logits[:, 1].numpy()
        # skip O and I label parsing
        for label_id in range(2, logits.shape[1]):
            s_o_indicator, predicate, s_o_type = id2label[label_id].split("-")
            label_logits = logits[:, label_id].numpy()
            parsing_results = parsing_entity(label_logits, label_I_logits)

            if parsing_results and s_o_indicator == "S":
                for parsing_result in parsing_results:
                    extract_subjects[predicate].append({"subject_ids":parsing_result, "subject_type":s_o_type})
            elif parsing_results and s_o_indicator == "O":
                for parsing_result in parsing_results:
                    extract_objects[predicate].append({"object_ids": parsing_result, "object_type": s_o_type})

        # convert result to spo format
        for predicate in extract_subjects.keys():
            if predicate not in extract_objects.keys():
                continue

            for subject_result in extract_subjects[predicate]:
                subject_ids, subject_type = subject_result["subject_ids"], subject_result["subject_type"]
                subject = example_text[subject_ids[0]:subject_ids[-1]+1]
                spo = {"predicate":predicate, "subject": subject, "subject_type":subject_type, "object":{}, "object_type":{}}
                for object_result in extract_objects[predicate]:
                    object_ids, object_type = object_result["object_ids"], object_result["object_type"]
                    object = example_text[object_ids[0]:object_ids[-1]+1]
                    object_type_keyname = get_object_keyname(reverse_schema, predicate, object_type)
                    spo["object"][object_type_keyname] = object
                    spo["object_type"][object_type_keyname] = object_type

                spo_list.append(spo)
        example = {"text": example_text, "spo_list":spo_list}
        batch_pred_examples.append(example)
    
    return batch_pred_examples
