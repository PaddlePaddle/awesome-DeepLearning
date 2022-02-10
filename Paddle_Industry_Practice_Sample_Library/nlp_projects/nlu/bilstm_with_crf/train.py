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
from data import ATISDataset
from paddle.io import DataLoader
from model import JointModel
from tqdm import tqdm
import paddle.nn.functional as F
from evaluate import evaluate

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


def train():
 
    # configuring model training
    with open("config.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read())

    train_set = ATISDataset(args["train_path"], args["vocab_path"], args["intent_path"], args["slot_path"])
    test_set = ATISDataset(args["test_path"], args["vocab_path"], args["intent_path"], args["slot_path"])

    print("train:",len(train_set))
    print("test:", len(test_set))
    args["vocab_size"] = train_set.vocab_size
    args["num_intents"] = train_set.num_intents
    args["num_slots"] = train_set.num_slots

    train_loader = DataLoader(train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True,  collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, drop_last=False, collate_fn=collate_fn)

    jointModel = JointModel(args["vocab_size"], args["embedding_size"], args["lstm_hidden_size"], args["num_intents"], args["num_slots"], num_layers=args["lstm_layers"], drop_p=args["dropout_rate"])

    use_gpu = True if paddle.get_device().startswith("gpu") else False
    if use_gpu:
        paddle.set_device('gpu:0')

    optimizer = paddle.optimizer.Adam(learning_rate=args["learning_rate"], beta1=0.9, beta2=0.99, parameters=jointModel.parameters())
        
    jointModel.train()
    # training and evaluating model
    for epoch in range(1, args["num_epochs"]+1): 
        for step, batch in enumerate(train_loader()):
            batch_tokens, batch_tags, batch_intents, batch_lens = batch
            emissions, intent_logits = jointModel(batch_tokens, batch_lens)
            # compute slot prediction loss
            slot_loss = jointModel.get_slot_loss(emissions, batch_lens, batch_tags)
            # compute intent prediction loss
            intent_loss = jointModel.get_intent_loss(intent_logits, batch_intents)
            # sum slot_loss and intent_loss 
            loss = slot_loss + intent_loss
        
            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()
            
            if step!=0 and step % args["log_steps"]  == 0:
                print("Epoch: %d, step: %d, total loss: %.4f, intent_loss: %.4f, slot_loss:%.4f" % (epoch, step, loss, intent_loss, slot_loss))
            if step!=0 and step % args["eval_steps"] == 0:
                evaluate(jointModel, test_set, args)
                jointModel.train()
     
        if (args["save_epochs"] != -1 and epoch % args["save_epochs"] == 0) or epoch == args["num_epochs"]:
            if not os.path.exists(args["save_dir"]):
                os.makedirs(args["save_dir"])
            save_model_path = os.path.join(args["save_dir"], "jointModel_e{}.pdparams".format(epoch))
            paddle.save(jointModel.state_dict(), save_model_path)   
 
    # save training args        
    save_args_path = os.path.join(args["save_dir"], "args.pdparams")
    paddle.save(args, save_args_path)


if __name__=="__main__":
    train()
