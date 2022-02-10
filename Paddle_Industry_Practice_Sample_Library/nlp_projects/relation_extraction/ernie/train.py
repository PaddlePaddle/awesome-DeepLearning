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
import copy
import argparse
import warnings
import numpy as np
from functools import partial
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import ErnieTokenizer, ErnieModel, LinearDecayWithWarmup
from evaluate import evaluate
from model import ErnieForTokenClassification
from utils.utils import set_seed
from utils.metric import SPOMetric
from data import read, generate_dict, load_dict, load_schema, load_reverse_schema, convert_example_to_feature

warnings.filterwarnings("ignore")

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--train_path", type=str, default=None, help="train data")
parser.add_argument("--dev_path", type=str, default=None, help="dev data")
parser.add_argument("--ori_schema_path", type=str, default=None, help="schema path")
parser.add_argument("--save_label_path", type=str, default=None, help="schema path")
parser.add_argument("--save_schema_path", type=str, default=None, help="schema path")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay rate for L2 regularizer.")
parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Warmup proportion params for warmup strategy")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max grad norm to clip gradient.")
parser.add_argument("--eval_step", type=int, default=500, help="evaluation step")
parser.add_argument("--log_step", type=int, default=50, help="log step")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def collate_fn(batch, pad_default_token_id=0):
    input_ids_list, token_type_ids_list, attention_mask_list, seq_len_list, labels_list = [], [], [], [], []
    for input_ids, token_type_ids, attention_mask, seq_len, labels in batch:
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        attention_mask_list.append(attention_mask)
        seq_len_list.append(seq_len)
        labels_list.append(labels)

    # padding sequence
    max_len = max(seq_len_list)
    for idx in range(len(input_ids_list)):
        pad_len = max_len - seq_len_list[idx]
        input_ids_list[idx] = input_ids_list[idx] + [pad_default_token_id] * pad_len
        token_type_ids_list[idx] = token_type_ids_list[idx] + [0] * pad_len
        attention_mask_list[idx] = attention_mask_list[idx] + [0] * pad_len   
      
        pad_label = labels_list[idx][0][:] # CLS label
        labels_list[idx] = labels_list[idx] + [pad_label] * pad_len


    return paddle.to_tensor(input_ids_list), paddle.to_tensor(token_type_ids_list), paddle.to_tensor(attention_mask_list), paddle.to_tensor(seq_len_list), paddle.to_tensor(labels_list)


class DuIELoss(paddle.nn.Layer):
    def __init__(self):
        super(DuIELoss, self).__init__()
        self.criterion = paddle.nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, logits, labels, masks):
        labels = paddle.cast(labels, "float32")
        loss = self.criterion(logits, labels)
        mask = paddle.cast(masks, "float32")
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        
        return loss.mean()


def train():
    # set running envir
    paddle.set_device(args.device)
    set_seed(args.seed)

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    model_name = "ernie-1.0"

    # load and process data
    label2id, id2label = load_dict(args.save_label_path)
    reverse_schema = load_reverse_schema(args.save_schema_path)

    train_ds = load_dataset(read, data_path=args.train_path, lazy=False)
    dev_ds = load_dataset(read, data_path=args.dev_path, lazy=False)
    dev_examples = copy.deepcopy(dev_ds)

    tokenizer = ErnieTokenizer.from_pretrained(model_name)
    trans_func = partial(convert_example_to_feature, tokenizer=tokenizer, label2id=label2id,  pad_default_label="O", max_seq_len=args.max_seq_len)

    train_ds = train_ds.map(trans_func, lazy=False)
    dev_ds = dev_ds.map(trans_func, lazy=False)
    
    # Warning: you should not set shuffle of dev_batch_sampler be True
    train_batch_sampler = paddle.io.BatchSampler(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=args.batch_size, shuffle=False)
    train_loader = paddle.io.DataLoader(dataset=train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_loader = paddle.io.DataLoader(dataset=dev_ds, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    # configure model training
    ernie = ErnieModel.from_pretrained(model_name)
    model = ErnieForTokenClassification(ernie, num_classes=len(label2id))

    num_training_steps = len(train_loader) * args.num_epoch
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    grad_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)
    optimizer = paddle.optimizer.AdamW(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=args.weight_decay, apply_decay_param_fun=lambda x: x in decay_params, grad_clip=grad_clip)

    criterion = DuIELoss()
    metric = SPOMetric()
    # start to train joint_model
    global_step, best_f1 = 0, 0.
    model.train()
    for epoch in range(1, args.num_epoch+1):
        for idx, batch_data in enumerate(train_loader()):
            input_ids, token_type_ids, attention_masks, seq_lens, labels = batch_data
            logits = model(input_ids, token_type_ids=token_type_ids)

            loss = criterion(logits, labels, attention_masks)

            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

            if global_step > 0 and global_step % args.log_step == 0:
                print(f"epoch: {epoch} - global_step: {global_step}/{num_training_steps} - loss:{loss.numpy().item():.6f}")
            if global_step > 0 and global_step % args.eval_step == 0:
                precision, recall, f1  = evaluate(model, dev_loader,  metric, dev_examples, reverse_schema, id2label, args.batch_size)
                model.train()
                if f1 > best_f1:
                    print(f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}")
                    best_f1 = f1
                    paddle.save(model.state_dict(), f"{args.checkpoint}/best.pdparams")
                print(f'evalution result: precision: {precision:.5f}, recall: {recall:.5f},  F1: {f1:.5f}')

            global_step += 1

    paddle.save(model.state_dict(), f"{args.checkpoint}/final.pdparams")

 
if __name__=="__main__":
    # generate label and schema dict
    generate_dict(args.ori_schema_path, args.save_label_path, args.save_schema_path)
    # train model
    train()


