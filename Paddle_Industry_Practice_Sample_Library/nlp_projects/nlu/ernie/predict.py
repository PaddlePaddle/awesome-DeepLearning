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
from paddlenlp.data import Stack, Pad, Tuple

from model import JointModel
from data import read, load_dict, convert_example_to_feature
from metric import SeqEntityScore, MultiLabelClassificationScore
from seqeval.metrics.sequence_labeling import get_entities


def predict(joint_model, tokenizer, id2intent, id2slot):

    joint_model.eval()
    while True:
        input_text = input("input text: \n")
        if input_text == "quit":
            break
        splited_input_text = list(input_text)
        features = tokenizer(splited_input_text, is_split_into_words=True, max_seq_len=args.max_seq_len, return_length=True)
        input_ids = paddle.to_tensor(features["input_ids"]).unsqueeze(0)
        token_type_ids = paddle.to_tensor(features["token_type_ids"]).unsqueeze(0)
        seq_len = features["seq_len"]

        history_ids = paddle.to_tensor(tokenizer("")["input_ids"]).unsqueeze(0)
        intent_logits, slot_logits = joint_model(input_ids, token_type_ids=token_type_ids, history_ids=history_ids)
        # parse intent labels
        intent_labels = [id2intent[idx] for idx, v in enumerate(intent_logits.numpy()[0]) if v > 0]
        
        # parse slot labels
        slot_pred_labels = slot_logits.argmax(axis=-1).numpy()[0][1:(seq_len)-1]
        slot_labels = []
        for idx in slot_pred_labels:
            slot_label = id2slot[idx]
            if slot_label != "O":
                slot_label = list(id2slot[idx])
                slot_label[1] = "-"
                slot_label = "".join(slot_label)
            slot_labels.append(slot_label)
        slot_entities = get_entities(slot_labels)
        
        # print result
        if intent_labels:
            print("intents: ", ",".join(intent_labels))
        else:
            print("intents: ", "无")
        for slot_entity in slot_entities:
            entity_name, start, end = slot_entity
            print(f"{entity_name}: ", "".join(splited_input_text[start:end+1]))



if __name__=="__main__":
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="model path that you saved")
    parser.add_argument("--slot_dict_path", type=str, default=None, help="slot dict path")
    parser.add_argument("--intent_dict_path", type=str, default=None, help="intent dict path")
    parser.add_argument("--use_history", type=bool, default=False, help="use history or not")    
    parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
    args = parser.parse_args()
    # yapf: enbale

    # load dev data
    model_name = "ernie-1.0"
    intent2id, id2intent = load_dict(args.intent_dict_path)
    slot2id, id2slot = load_dict(args.slot_dict_path)
    
    tokenizer = ErnieTokenizer.from_pretrained(model_name)

    # load model
    loaded_state_dict = paddle.load(args.model_path)
    ernie = ErnieModel.from_pretrained(model_name)
    joint_model = JointModel(ernie, len(slot2id), len(intent2id), use_history=args.use_history, dropout=0.1)    
    joint_model.load_dict(loaded_state_dict)

 
    # evalute on dev data
    predict(joint_model, tokenizer, id2intent, id2slot) 


