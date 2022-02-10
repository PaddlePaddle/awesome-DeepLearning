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
import argparse
import paddle
from paddle.io import DataLoader
from metric import SeqEntityScore
from model import JointModel

parser = argparse.ArgumentParser(description="processing input params.")
metric = SeqEntityScore


def load_dict(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        dict2id = dict(zip(words, range(len(words))))
        id2dict = {v:k for k,v in dict2id.items()}

        return dict2id, id2dict



def predict(jointModel, token2id, id2slot, metric):
    jointModel.eval()
 
    while True:
        input_text = input("input query: ")
        if input_text == "quit":
            break

        splited_text = input_text.split()
        tokens = [token2id.get(token, token2id["[unk]"]) for token in splited_text]
        tokens_len = len(tokens)
       
        if tokens_len < 2:
            print(f"the squence [{input_text}] is too short, please input valid text sequence.") 
            continue
        
        # constructing data to input to model
        tokens = paddle.to_tensor(tokens, dtype="int64").unsqueeze(0)
        tokens_len = paddle.to_tensor([tokens_len], dtype="int64")

        # computing emission score and intent score
        emissions, intent_logits = jointModel(tokens, tokens_len)
        
        # decoding with viterbi
        _, pred_paths = jointModel.viterbi_decoder(emissions, tokens_len)
        entities = metric.get_entities_bio(pred_paths[0][:tokens_len[0]]) 
        
        # obtaining the intent
        intent_id = paddle.argmax(intent_logits, axis=1).numpy()[0]

        # printing result
        print("intent:", id2intent[intent_id])
        for entity in entities:
            entity_type, start, end = entity
            entity_text = " ".join(splited_text[start:end+1])
            print(f"{entity_text} : {entity_type}")
    


if __name__=="__main__":
    parser.add_argument("--model_path", default="", help="the path of the saved model that you would like to verify")
    model_path = parser.parse_args().model_path
    
    # configuring model training
    with open("config.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read())

    # load token and slot dict
    token2id, id2token = load_dict(args["vocab_path"])
    intent2id, id2intent = load_dict(args["intent_path"])
    slot2id, id2slot = load_dict(args["slot_path"])
    args["vocab_size"] = len(token2id)
    args["num_intents"] = len(intent2id)
    args["num_slots"] = len(slot2id)

    metric = SeqEntityScore(id2slot)

    # load model
    loaded_state_dict = paddle.load(model_path)
    jointModel = JointModel(args["vocab_size"], args["embedding_size"], args["lstm_hidden_size"], args["num_intents"], args["num_slots"], num_layers=args["lstm_layers"], drop_p=args["dropout_rate"])
    jointModel.load_dict(loaded_state_dict)


    predict(jointModel, token2id, id2slot, metric)
