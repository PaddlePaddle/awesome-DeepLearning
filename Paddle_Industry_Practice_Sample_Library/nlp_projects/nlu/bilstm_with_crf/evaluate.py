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
import yaml
import paddle
import argparse
from metric import SeqEntityScore, SingleClassificationScore
from paddle.io import DataLoader
from model import JointModel
from data import ATISDataset

parser = argparse.ArgumentParser(description="processing input prams.")

def collate_fn(batch, token_pad_val=0, tag_pad_val=0):
    token_list, tag_list, intent_list, len_list = [], [], [], []
    for tokens, tags, intent, len_ in batch:
        assert len(tokens) == len(tags)
        token_list.append(tokens.tolist())
        tag_list.append(tags.tolist())
        intent_list.append(intent)
        len_list.append(len_)
    # padding sequence
    max_len = max(map(len, token_list))
    for i in range(len(token_list)):
        token_list[i] = token_list[i] + [token_pad_val] * (max_len-len(token_list[i]))
        tag_list[i] = tag_list[i] + [tag_pad_val] * (max_len - len(tag_list[i]))

    return paddle.to_tensor(token_list), paddle.to_tensor(tag_list), paddle.to_tensor(intent_list), paddle.to_tensor(len_list)

def evaluate(jointModel=None, test_set=None, args=None):
    jointModel.eval()

    test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, drop_last=False, collate_fn=collate_fn)
    slot_metric = SeqEntityScore(test_set.id2slot)
    intent_metric = SingleClassificationScore(test_set.id2intent)
    
    for step, batch in enumerate(test_loader()):
        batch_tokens, batch_tags, batch_intents, batch_lens = batch
        emissions, intent_logits = jointModel(batch_tokens, batch_lens)
        _, pred_paths = jointModel.viterbi_decoder(emissions, batch_lens)
        
        pred_paths = pred_paths.numpy().tolist()
        pred_paths = [tag_seq[:tag_len] for tag_seq, tag_len in zip(pred_paths, batch_lens)]
        
        batch_tags = batch_tags.numpy().tolist()
        real_paths = [tag_seq[:tag_len] for tag_seq, tag_len in zip(batch_tags, batch_lens)]
        slot_metric.update(pred_paths=pred_paths, real_paths=real_paths)
         
        pred_intents = paddle.argmax(intent_logits, axis=1)
        intent_metric.update(pred_intents, batch_intents)

    print("\n================evaluate result================")
    intent_result = intent_metric.get_result()
    slot_result = slot_metric.get_result()
    intent_metric.format_print(intent_result)
    slot_metric.format_print(slot_result)
    print("\n")


if __name__=="__main__":
    parser.add_argument("--model_path", type=str, default="", help="the path of the saved model that you would like to verify")
    model_path = parser.parse_args().model_path

    # configuring model training
    with open("config.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read())

    # loading testset
    test_set = ATISDataset(args["test_path"], args["vocab_path"], args["intent_path"], args["slot_path"])
    args["vocab_size"] = test_set.vocab_size
    args["num_intents"] = test_set.num_intents
    args["num_slots"] = test_set.num_slots

    # loading model
    loaded_state_dict = paddle.load(model_path)
    jointModel = JointModel(args["vocab_size"], args["embedding_size"], args["lstm_hidden_size"], args["num_intents"], args["num_slots"], num_layers=args["lstm_layers"], drop_p=args["dropout_rate"])
    jointModel.load_dict(loaded_state_dict)

    # evaluate model
    evaluate(jointModel, test_set, args)
